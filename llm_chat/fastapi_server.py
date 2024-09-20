#!/usr/bin/env python
import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional, Literal, get_args

import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from tensorrt_llm.executor import ExecutorBindingsWorker
from tensorrt_llm.hlapi import LLM, BuildConfig, KvCacheConfig, SamplingParams
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from tensorrt_llm.hlapi.llm_utils import QuantConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from llm_chat.openai.protocol import CompletionRequest, CompletionResponse, ErrorResponse
from llm_chat.serving_engine import OpenAIServingCompletion

# docs
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_GRACEFUL_SHUTDOWN = 15  # seconds.

VALID_QUANT_ALGOS = Literal[f"{QuantAlgo.W8A16}", f"{QuantAlgo.W4A16}",
                            f"{QuantAlgo.W4A16_AWQ}", f"{QuantAlgo.W4A8_AWQ}",
                            f"{QuantAlgo.W4A16_GPTQ}", f"{QuantAlgo.FP8}",
                            f"{QuantAlgo.INT8}"]

def _wait_and_return_queue(self, req_id: int):
    """Monkey patch ExecutorBindingsWorker.return_queue to avoid req_id key not found error."""
    if self.result_queue is not None:
        return self.result_queue
    start_time = time.time()
    while time.time() - start_time <= 1:  # 1 second timeout
        if req_id in self._results:
            return self._results[req_id].queue
        time.sleep(0.01)  # wait for 10ms
    raise TimeoutError(f"Timeout waiting for req_id {req_id} to be in _results")


ExecutorBindingsWorker.return_queue = _wait_and_return_queue

class LlmServer:

    def __init__(self, llm: LLM, kv_cache_config: KvCacheConfig):
        self.llm = llm
        self.kv_cache_config = kv_cache_config
        self.openai_serving_completion = OpenAIServingCompletion(self.llm)

        self.app = FastAPI()
        self.register_routes()

    def register_routes(self):
        self.app.add_api_route("/stats", self.stats, methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/generate", self.generate, methods=["POST"])
        self.app.add_api_route("/v1/completions", self.create_completion, methods=["POST"])

    async def stats(self) -> Response:
        content = await self.llm._executor.aget_stats()
        return JSONResponse(json.loads(content))

    async def health(self) -> Response:
        return Response(status_code=200)

    async def generate(self, request: Request) -> Response:
        ''' Generate completion for the request.

        The request should be a JSON object with the following fields:
        - prompt: the prompt to use for the generation.
        - stream: whether to stream the results or not.
        - other fields: the sampling parameters (See `SamplingParams` for details).
        '''
        request_dict = await request.json()

        prompt = request_dict.pop("prompt", "")
        streaming = request_dict.pop("streaming", False)

        sampling_params = SamplingParams(**request_dict)

        promise = self.llm.generate_async(prompt,
                                          streaming=streaming,
                                          sampling_params=sampling_params)

        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for output in promise:
                yield output.outputs[0].text_diff.encode("utf-8")

        if streaming:
            return StreamingResponse(stream_results())

        # Non-streaming case
        await promise.aresult()
        return JSONResponse({"text": promise.outputs[0].text})

    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)
        elif isinstance(generator, CompletionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")

    async def __call__(self, host, port):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                                timeout_graceful_shutdown=TIMEOUT_GRACEFUL_SHUTDOWN)
        await uvicorn.Server(config).serve()


@click.command()
@click.argument("model_dir")
@click.option("--tokenizer", type=str, default=None)
@click.option("--host", type=str, default=None)
@click.option("--port", type=int, default=8000)
@click.option("--max_running", type=int, default=8)
# @click.option("--max_num_tokens", type=int, default=12)
@click.option("--max_seq_len", type=int, default=8192)
@click.option("--max_beam_width", type=int, default=1)
@click.option("--tp_size", type=int, default=1, help="The number of devices for tensor parallelism to use")
@click.option("--pp_size", type=int, default=1)
@click.option(
    "--quantization",
    "-q",
    type=click.Choice(tuple(get_args(VALID_QUANT_ALGOS))),
    default=None,
    help=
    ("The quantization algorithm to be used. See the "
     "documentations for more information.\n"
     "  - https://nvidia.github.io/TensorRT-LLM/precision.html"
     "  - https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md"
     ),
)
def entrypoint(model_dir: str,
               tokenizer: Optional[str] = None,
               host: Optional[str] = None,
               port: int = 8000,
               max_running: int = 8,
               max_seq_len: int = 1024,
               max_beam_width: int = 1,
               quantization: str = "W8A8",
               tp_size: int = 1,
               pp_size: int = 1):
    host = host or "0.0.0.0"
    port = port or 8000
    logging.info(f"Starting server at {host}:{port}")
    logger.set_level("info")

    build_config = BuildConfig(max_batch_size=max_running, max_beam_width=max_beam_width, max_seq_len=max_seq_len)

    # Set the compute quantization.
    quant_algo = QuantAlgo(quantization) if quantization is not None else None
    quant_config = QuantConfig()
    quant_config.quant_algo = quant_algo
    # If the quantization is FP8, force the KV cache dtype to FP8.
    quant_config.kv_cache_quant_algo = quant_algo.value \
         if quant_algo == QuantAlgo.FP8 else None

    # Enable FHMA, and FP8 FMHA if FP8 quantization is enabled.
    # TODO: Revisit, there is an issue with enabling FHMA. If only
    # paged FMHA is enabled with FP8 quantization, the Builder
    # will not enable the FP8 FMHA.
    build_config.plugin_config.use_paged_context_fmha = True
    build_config.plugin_config.use_fp8_context_fmha = True \
         if quant_algo == QuantAlgo.FP8 else False

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
    kv_cache_config.enable_block_reuse = True

    llm = LLM(model_dir,
              tokenizer,
              tensor_parallel_size=tp_size,
              pipeline_parallel_size=pp_size,
              dtype="auto",
              kv_cache_config=kv_cache_config,
              build_config=build_config,
              quant_config=quant_config)
    # llm.save(engine_dir)

    server = LlmServer(llm=llm, kv_cache_config=kv_cache_config)

    asyncio.run(server(host, port))


if __name__ == "__main__":
    entrypoint()
