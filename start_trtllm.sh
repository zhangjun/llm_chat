MODEL_PATH='/opt/trtllm/'
python3 llm_chat/fastapi_server.py  --tp_size 1 --pp_size 1 ${MODEL_PATH}