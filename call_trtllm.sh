
service_name=test_as_8b_trtllm
token='ZmNlZmI0YjVjNmE5ZTY3YWZmNjQ3MGZlOTYwOGVkMzQwOGY2ZGM1ZQ=='

# url="http://1893706806886638.cn-beijing.pai-eas.aliyuncs.com/api/predict/${service_name}/generate"
url="localhost:8000/generate"
url="localhost:8000/v1/completions"

# curl -v -X GET http://1893706806886638.cn-beijing.pai-eas.aliyuncs.com/api/predict/${service_name}/health -H "Authorization: Bearer ${token}"

curl -v -X POST -d \
    '{"prompt": "The capital of France is","max_tokens": 256,"temperature": 0.80,"stream": false, "top_p":1, "top_k":3, "repetition_penalty":1.08, "length_penalty":1.0, "model":"trtllm"}' \
    --header "Content-Type: application/json" $url \
    -H "Authorization: Bearer ${token}" -s -w %{time_total}
