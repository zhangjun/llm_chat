# run
HF_ENDPOINT=https://hf-mirror.com python3 quantize.py --model_dir /mnt/data/shared/test/aistory/8b_llama3_as_en_aug_v3_0807 --device cpu --qformat fp8 --tokenizer_max_seq_length 4096 --output_dir /mnt/data/zhangjun/mydev/trtllm_tp1pp1_0905 --tp_size 1 --pp_size 1 --kv_cache_dtype fp8

# setup
## env
```shell
TLLM_HLAPI_BUILD_CACHE=1
```
## LLmArgs
enable_build_cache

