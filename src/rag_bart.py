from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载预训练的生成模型
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
# tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
# model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

# 创建一些示例文档
documents = [
    "The Eiffel Tower is located in Paris.",
    "The capital of Japan is Tokyo.",
    "Python is a popular programming language.",
    "吴相铮是一名资深程序员，他最喜欢吃的食物是鸡蛋。",
    "吴相铮目前的工作地点是北京。",
    "Mount Fuji is the highest mountain in Japan."
]

# 文本检索部分
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

# 使用 faiss 创建索引
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_vectors.astype(np.float32))

def retrieve(query, k=2):
    # 将查询转换为向量
    query_vec = vectorizer.transform([query]).toarray().astype(np.float32)

    # 检索相关文档
    D, I = index.search(query_vec, k)
    return [documents[i] for i in I[0]]

# 生成部分
def generate(query, k=2):
    # 首先检索相关文档
    retrieved_docs = retrieve(query, k)
    print(retrieved_docs)

    # 将检索到的文档作为上下文拼接到查询中
    context = " ".join(retrieved_docs)
    input_text = f"Query: {query} Context: {context}"

    # 使用生成模型生成文本
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,   # 生成文本的最大长度
        min_length=40,    # 生成文本的最小长度
        length_penalty=2.0, # 控制生成文本长度的惩罚
        num_beams=4,      # 使用的束搜索的数量
        early_stopping=True  # 提前停止生成
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 示例查询
query = "吴相铮是谁"
output = generate(query)
print(output)
