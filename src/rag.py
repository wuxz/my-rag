from rag_ali import RagImpByAli

query = "吴相铮是谁？"
rag = RagImpByAli()
answer = rag.query(query)
print("生成的回答：", answer)
