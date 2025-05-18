import requests

# 本機 Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/chat"

# 建立訊息內容
data = {
    "model": "llama2",  # 替換為你已安裝的模型
    "messages": [
        {"role": "user", "content": "請用繁體中文解釋什麼是大型語言模型"}
    ]
}

# 發送 POST 請求
response = requests.post(OLLAMA_URL, json=data)

# 解析回應內容
if response.status_code == 200:
    result = response.json()
    print("\n🤖 回應內容：")
    print(result['message']['content'])
else:
    print("❌ 發生錯誤：", response.text)
