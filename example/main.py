import asyncio
import sys

from tensorrt_llm import LLM, SamplingParams

from tensorrt_llm.hlapi import LLM, BuildConfig, KvCacheConfig, SamplingParams
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from tensorrt_llm.hlapi.llm_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

tp_size, pp_size = 4, 2
# model could accept HF model name or a path to local HF model.
model_path = "/mnt/data/zhangjun/mydev/trtllm_tp1pp1_0905"
tokenizer_path = "/mnt/data/shared/test/aistory/8b_llama3_as_en_aug_v3_0807"
model_path = "/mnt/models/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79/"
tokenizer_path = "/mnt/models/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79/"
engine_path = f"/mnt/zhangjun/mydev/qwen72b_tp{tp_size}pp{pp_size}/"

max_running = 16     # max_batch_size
max_beam_width = 1
max_seq_len = 8192  # max_seq_len 512
build_config = BuildConfig(max_batch_size=max_running, max_beam_width=max_beam_width, max_seq_len=max_seq_len)

quantization = "FP8" # "FP8", None
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
llm = LLM(model=model_path,
    tokenizer=tokenizer_path,
    # enable_build_cache=True,
    tensor_parallel_size=tp_size,
    pipeline_parallel_size=pp_size,
    build_config=build_config,
    quant_config=quant_config)

build_engine = True
if build_engine:
    # save engine
    llm.save(engine_path)
    sys.exit(1)
# Sample prompts.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    "The future of AI is",
]

# Create a sampling params.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# Async based on Python coroutines
async def task(id: int, prompt: str):

    # streaming=True is used to enable streaming generation.
    # async for output in llm.generate_async(prompt,
    #                                        sampling_params,
    #                                        streaming=True):
    generator = llm.generate_async(prompt,
                                    sampling_params,
                                    streaming=True)
    async for output in generator:
        print(f"Generation for prompt-{id}: {output.outputs[0].text!r}")
        print(output)


async def main():
    tasks = [task(id, prompt) for id, prompt in enumerate(prompts)]
    await asyncio.gather(*tasks)


asyncio.run(main())
