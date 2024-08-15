import requests

# Hugging Face API 的配置信息
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_TOKEN = "your_huggingface_api_token"  # 在 Hugging Face 账户中生成 API Token

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# 文本检索部分（保持不变）
documents = [
    "The Eiffel Tower is located in Paris.",
    "The capital of Japan is Tokyo.",
    "Python is a popular programming language.",
    "Mount Fuji is the highest mountain in Japan.",
    "吴相铮是一名资深程序员，他最喜欢吃的食物是鸡蛋。",
    "吴相铮目前的工作地点是北京。"
]

# 简单的 TF-IDF 检索
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

def retrieve(query, k=2):
    query_vec = vectorizer.transform([query]).toarray()
    scores = np.dot(doc_vectors, query_vec.T).flatten()
    top_k_indices = np.argsort(scores)[-k:]
    return [documents[i] for i in top_k_indices]

# 生成部分使用 Hugging Face API
def generate(query, k=2):
    # 检索相关文档
    retrieved_docs = retrieve(query, k)
    context = " ".join(retrieved_docs)
    input_text = f"Query: {query} Context: {context}"

    # 使用 Hugging Face API 生成文本
    payload = {"inputs": input_text}
    output = query_huggingface(payload)

    # 处理 API 返回结果
    if isinstance(output, list):
        return output[0]["summary_text"]
    else:
        return "Error: Unable to generate text."

# 示例查询
query = "Where is Mount Fuji?"
output = generate(query)
print(output)
