from typing import AsyncGenerator, AsyncIterator, Optional, Literal, get_args, TypedDict, Sequence, Type, Iterator, cast
from pydantic import Field
from typing_extensions import Annotated, TypeIs, assert_never
import uuid
from http import HTTPStatus
from fastapi import Request

from tensorrt_llm.hlapi import LLM
from tensorrt_llm.hlapi.llm import RequestOutput
from llm_chat.utils import *
from llm_chat.openai.protocol import *

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def is_list_of(
    value: object,
    typ: Type[T],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[List[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)

class ParsedText(TypedDict):
    content: str
    is_tokens: Literal[False]

class ParsedTokens(TypedDict):
    content: List[int]
    is_tokens: Literal[True]

def parse_and_batch_prompt(
    prompt: Union[str, List[str], List[int], List[List[int]]],
) -> Union[Sequence[ParsedText], Sequence[ParsedTokens]]:
    if isinstance(prompt, str):
        return [ParsedText(content=prompt, is_tokens=False)]
    
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")

        if is_list_of(prompt, str):
            # case 2: array of strings
            return [
                elem for elem in prompt
            ]
        if is_list_of(prompt, int):
            # case 3: array of tokens
            return [ParsedTokens(content=prompt, is_tokens=True)]
        if is_list_of(prompt, list):
            if len(prompt[0]) == 0:
                raise ValueError("please provide at least one prompt")

            if is_list_of(prompt[0], int):
                # case 4: array of token arrays
                return [
                    ParsedTokens(content=elem, is_tokens=True)
                    for elem in prompt
                ]

    raise ValueError("prompt must be a string, array of strings, "
                     "array of tokens, or array of token arrays")

AnyRequest = Union[CompletionRequest, DetokenizeRequest,
                   TokenizeRequest]

class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: List[int]

class OpenAIServing:
    def __init__(self, llm_engine: LLM):
        super().__init__()
        self.llm_engine = llm_engine

    def _tokenize_prompt_input_or_inputs(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        input_or_inputs: Union[str, List[str], List[int], List[List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[TextTokensPrompt]:
        """
        Tokenize/detokenize depending on the input format.

        According to `OpenAI API <https://platform.openai.com/docs/api-reference/embeddings/create>`_
        , each input can be a string or array of tokens. Note that each request
        can pass one or more inputs.
        """
        for prompt_input in parse_and_batch_prompt(input_or_inputs):
            # Although our type checking is based on mypy,
            # VSCode Pyright extension should still work properly
            # "is True" is required for Pyright to perform type narrowing
            # See: https://github.com/microsoft/pyright/issues/7672
            if prompt_input["is_tokens"] is False:
                yield self._normalize_prompt_text_to_input(
                    request,
                    tokenizer,
                    prompt=prompt_input["content"],
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield self._normalize_prompt_tokens_to_input(
                    request,
                    tokenizer,
                    prompt_ids=prompt_input["content"],
                )
        
    def _normalize_prompt_text_to_input(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt: str,
        # truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
        add_special_tokens: bool,
    ) -> TextTokensPrompt:
        encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        # if truncate_prompt_tokens is None:
        #     encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        # else:
        #     encoded = tokenizer(prompt,
        #                         add_special_tokens=add_special_tokens,
        #                         truncation=True,
        #                         max_length=truncate_prompt_tokens)

        input_ids = encoded.input_ids

        input_text = prompt

        return self._validate_input(request, input_ids, input_text)

    def _normalize_prompt_tokens_to_input(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt_ids: List[int],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    ) -> TextTokensPrompt:
        input_ids = prompt_ids
        # if truncate_prompt_tokens is None:
        #     input_ids = prompt_ids
        # else:
        #     input_ids = prompt_ids[-truncate_prompt_tokens:]

        input_text = tokenizer.decode(input_ids)

        return self._validate_input(request, input_ids, input_text)

    def _validate_input(
        self,
        request: AnyRequest,
        input_ids: List[int],
        input_text: str,
    ) -> TextTokensPrompt:
        token_num = len(input_ids)

        if request.max_tokens is None:
            if token_num >= self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages, "
                    f"Please reduce the length of the messages.")
        elif token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.")

        return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

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
        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())
        generators: List[AsyncGenerator[RequestOutput, None]] = []
        try:
            # prompts = list(
            #     self._tokenize_prompt_input_or_inputs(
            #         request,
            #         tokenizer,
            #         request.prompt,
            #         add_special_tokens=request.add_special_tokens,
            #     ))
            prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
            for i, prompt_inputs in enumerate(prompts):
                request_id_item = f"{request_id}-{i}"
                sampling_params = request.to_sampling_params(
                      default_max_tokens=1024)
                # sampling_params = SamplingParams(**request_dict)
                # sampling_params = request.to_sampling_params(
                #     default_max_tokens=self.max_model_len -
                #     len(prompt_inputs["prompt_token_ids"]))
                # sampling_params = SamplingParams()
                # sampling_params.beam_width = 1
                # sampling_params.max_new_tokens = 100
                # sampling_params.temperature = 0.5
                # sampling_params.top_p = 0.95

                generator = self.llm_engine.generate_async(prompt_inputs,
                                                streaming=request.stream,
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
                                                    num_prompts=len(prompts))

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
                    # finish_reason = output.finish_reason
                    # stop_reason = output.stop_reason

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                # logprobs=logprobs,
                                # finish_reason=finish_reason,
                                # stop_reason=stop_reason,
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
                    # finish_reason=output.finish_reason,
                    # stop_reason=output.stop_reason,
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

