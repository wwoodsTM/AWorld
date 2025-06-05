from openai import OpenAI

# 方法1: 使用自定义API地址（如本地部署的模型）
client = OpenAI(
    api_key="sk-9329256ff1394003b6761615361a8f0f",  # 替换为你的API密钥
    base_url="https://agi.alipay.com/api"  # 默认OpenAI地址，可替换为其他地址
)

# 发送请求 - 可以指定不同的模型
response = client.chat.completions.create(
    model="shangshu.claude-3.7-sonnet",  # 可替换为其他模型名称
    messages=[
        {"role": "user", "content": "杭州如何"}
    ],
    temperature=0.7,  # 可选：控制回复的随机性
    max_tokens=1000   # 可选：限制回复长度
)

if __name__ == '__main__':
    # 打印回复
    print(response.choices[0].message.content)
