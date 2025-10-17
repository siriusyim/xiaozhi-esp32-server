from openai import OpenAI
import litellm
# 创建OpenAI客户端对象
client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="dummy"  # 任意字符串即可
)

# 调用 Qwen
resp = client.chat.completions.create(
    model="qwen-turbo",
    messages=[{"role": "user", "content": "用中文解释量子计算"}]
)
print(f"=====> m1: {resp.choices[0].message.content}")


# 方法2: 直接调用 LiteLLM (需要确保模型名称包含提供商信息)
resp = litellm.completion(
    model="dashscope/qwen-turbo",  # 注意: 直接调用时需要使用完整的模型名称，包含提供商信息
    messages=[{"role": "user", "content": "用中文解释量子计算"}],
    api_key="xxxx",  # 直接调用时需要提供API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 直接调用时需要提供基础URL
)
print(f"=====> m2: {resp.choices[0].message.content}")
