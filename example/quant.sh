tp_size=4
pp_size=2
model_dir="/mnt/models/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79/"
output_dir="/mnt/zhangjun/mydev/qwen72b_tp${tp_size}pp${pp_size}/"
HF_ENDPOINT=https://hf-mirror.com python3 quantize.py --model_dir ${model_dir} --device cuda --qformat fp8 --tokenizer_max_seq_length 8192 --output_dir ${output_dir} --tp_size 4 --pp_size 2 --kv_cache_dtype fp8
