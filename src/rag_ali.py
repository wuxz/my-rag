from http import HTTPStatus
import dashscope
from transformers import AutoTokenizer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载预训练的生成模型
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

documents = [
    "The Eiffel Tower is located in Paris.",
    "The capital of Japan is Tokyo.",
    "Python is a popular programming language.",
    "吴相铮是一名资深程序员，他最喜欢吃的食物是鸡蛋。",
    "吴相铮目前的工作地点是北京。",
    "Mount Fuji is the highest mountain in Japan."
]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

# 使用 faiss 创建索引
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_vectors.astype(np.float32))

class RagImpByAli:
    # 用您的 DashScope API-KEY 代替 YOUR_DASHSCOPE_API_KEY
    # export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"

    # 检索函数（示例中假设你已经有一个检索系统）
    def __retrieve_documents(self, query, k=2):
        # 将查询转换为向量
        query_vec = vectorizer.transform([query]).toarray().astype(np.float32)

        # 检索相关文档
        D, I = index.search(query_vec, k)
        return [documents[i] for i in I[0]]

    # 生成回答的函数
    def __generate_answer(self, query, documents):
        # 组合用户查询和检索到的文档
        context = " ".join(documents)
        prompt = f"用户查询：{query}\n相关文档：{context}\n回答："

        response = dashscope.Generation.call(
            model="qwen-turbo",
            prompt=prompt
        )
        # 如果调用成功，则打印模型的输出
        if response.status_code == HTTPStatus.OK:
            answer = response.output.text
        # 如果调用失败，则打印出错误码与失败信息
        else:
            answer = f'发生错误, code:{response.code}, message:{response.message}'

        return answer

    def query(self, query):
        return self.__generate_answer(query, self.__retrieve_documents(query))
