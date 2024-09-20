# llm_chat

## install
```shell
git clone https://github.com/zhangjun/llm_chat
cd llm_chat
# https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html
python -m build
pip instal dist/llmchat-*.whl
```
## usage
```shell
python -m llm_chat.serve --tp_size 1 --pp_size 1 ${MODEL_PATH}
```