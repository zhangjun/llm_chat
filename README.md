# llm_chat

## install
```shell
git clone https://github.com/zhangjun/llm_chat
cd llm_chat
python setup.py bdist_wheel
pip instal dist/llmchat-*.whl
```
## usage
```shell
python -m llm_chat.serve --tp_size 1 --pp_size 1 ${MODEL_PATH}
```