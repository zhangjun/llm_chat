from typing import AsyncGenerator, Optional, Literal, get_args
import uuid
from http import HTTPStatus

from tensorrt_llm.hlapi import LLM
from llm_chat.utils import *
from llm_chat.protocol import *

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

class OpenAIServing:
    def __init__(llm_engine: LLM):
        super().__init__()
        self.llm_engine = llm_engine

class OpenAIServingCompletion(OpenAIServing):
    def __init__(self, llm_engine: LLM):
        super().__init__(llm_engine = llm_engine)

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error":
            self.create_error_response(message=message,
                                       err_type=err_type,
                                       status_code=status_code).model_dump()
        })
        return json_str
    
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())
        generators: List[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, prompt_inputs in enumerate(prompts):
                request_id_item = f"{request_id}-{i}"
                generator = self.llm.generate_async(prompt_inputs["prompt_token_ids"],
                                                streaming=srequest.stream,
                                                sampling_params=sampling_params)
                # generator = self.llm_engine.generate(
                #     {"prompt_token_ids": prompt_inputs["prompt_token_ids"]},
                #     sampling_params,
                #     request_id_item,
                # )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(
            *generators, is_cancelled=raw_request.is_disconnected)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. In addition, we do not stream the results when use
        # beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        # Streaming response
        if stream:
            return self.completion_stream_generator(request,
                                                    result_generator,
                                                    request_id,
                                                    created_time,
                                                    model_name,
                                                    num_prompts=len(prompts),
                                                    tokenizer=tokenizer)

        # Non-streaming response
        final_res_batch: List[Optional[RequestOutput]] = [None] * len(prompts)
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    final_res.prompt = prompts[i]["prompt"]

            final_res_batch_checked = cast(List[RequestOutput],
                                           final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response        
        
    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        result_generator: AsyncIterator[Tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_texts = [""] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts

        try:
            async for prompt_idx, res in result_generator:
                prompt_token_ids = res.prompt_token_ids
                # prompt_logprobs = res.prompt_logprobs
                prompt_text = res.prompt

                delta_token_ids: GenericSequence[int]
                # out_logprobs: Optional[GenericSequence[Optional[Dict[
                #     int, Logprob]]]]

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices
                    # TODO(simon): optimize the performance by avoiding full
                    # text O(n^2) sending.

                    assert request.max_tokens is not None
                    if request.echo and request.max_tokens == 0:
                        assert prompt_text is not None
                        # only return the prompt
                        delta_text = prompt_text
                        delta_token_ids = prompt_token_ids
                        # out_logprobs = prompt_logprobs
                        has_echoed[i] = True
                    elif (request.echo and request.max_tokens > 0
                          and not has_echoed[i]):
                        # assert prompt_text is not None
                        # assert prompt_logprobs is not None
                        # echo the prompt and first token
                        delta_text = prompt_text + output.text
                        delta_token_ids = [
                            *prompt_token_ids, *output.token_ids
                        ]
                        # out_logprobs = [
                        #     *prompt_logprobs,
                        #     *(output.logprobs or []),
                        # ]
                        has_echoed[i] = True
                    else:
                        # return just the delta
                        delta_text = output.text[len(previous_texts[i]):]
                        delta_token_ids = output.token_ids[
                            previous_num_tokens[i]:]
                        # out_logprobs = output.logprobs[previous_num_tokens[
                        #     i]:] if output.logprobs else None

                    logprobs = None

                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                # logprobs=logprobs,
                                finish_reason=finish_reason,
                                stop_reason=stop_reason,
                            )
                        ])
                    if (request.stream_options
                            and request.stream_options.include_usage):
                        if (request.stream_options.continuous_usage_stats
                                or output.finish_reason is not None):
                            prompt_tokens = len(prompt_token_ids)
                            completion_tokens = len(output.token_ids)
                            usage = UsageInfo(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=prompt_tokens + completion_tokens,
                            )
                        if request.stream_options.continuous_usage_stats:
                            chunk.usage = usage
                        else:
                            chunk.usage = None

                    response_json = chunk.model_dump_json(exclude_unset=False)
                    yield f"data: {response_json}\n\n"

            if (request.stream_options
                    and request.stream_options.include_usage):
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=usage,
                )
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=False, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: List[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: AnyTokenizer,
    ) -> CompletionResponse:
        choices: List[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0

        for final_res in final_res_batch:
            prompt_token_ids = final_res.prompt_token_ids
            # prompt_logprobs = final_res.prompt_logprobs
            prompt_text = final_res.prompt

            token_ids: GenericSequence[int]
            # out_logprobs: Optional[GenericSequence[Optional[Dict[int,
            #                                                      Logprob]]]]

            for output in final_res.outputs:
                assert request.max_tokens is not None
                if request.echo and request.max_tokens == 0:
                    assert prompt_text is not None
                    token_ids = prompt_token_ids
                    # out_logprobs = prompt_logprobs
                    output_text = prompt_text
                elif request.echo and request.max_tokens > 0:
                    assert prompt_text is not None
                    token_ids = [*prompt_token_ids, *output.token_ids]

                    # if request.logprobs is None:
                    #     out_logprobs = None
                    # else:
                    #     assert prompt_logprobs is not None
                    #     assert output.logprobs is not None
                    #     out_logprobs = [
                    #         *prompt_logprobs,
                    #         *output.logprobs,
                    #     ]

                    output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    # out_logprobs = output.logprobs
                    output_text = output.text

                # if request.logprobs is not None:
                #     assert out_logprobs is not None, "Did not output logprobs"
                #     logprobs = self._create_completion_logprobs(
                #         token_ids=token_ids,
                #         top_logprobs=out_logprobs,
                #         tokenizer=tokenizer,
                #         num_output_top_logprobs=request.logprobs,
                #     )
                # else:
                #     logprobs = None

                choice_data = CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    # logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    # prompt_logprobs=final_res.prompt_logprobs,
                )
                choices.append(choice_data)

            num_prompt_tokens += len(prompt_token_ids)
            num_generated_tokens += sum(
                len(output.token_ids) for output in final_res.outputs)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

