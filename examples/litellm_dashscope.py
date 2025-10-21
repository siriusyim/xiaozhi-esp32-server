from litellm import completion
import os

response = completion(
    model="dashscope/qwen-turbo", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    messages=[
       {"role": "user", "content": "你是谁？"}
   ],
)
print(response)
