from flask import Flask, request, jsonify, Response
from rag_ali import RagImpByAli

# Flask 应用程序初始化
app = Flask(__name__)
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'
app.config['JSON_AS_ASCII'] = False

@app.after_request
def add_header(response):
    if response.headers['Content-Type'] == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

@app.route('/query', methods=['POST', 'GET'])
def query():
    data = request.json
    query_text = data.get('query')

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    rag = RagImpByAli()
    answer = rag.query(query_text)
    print("生成的回答：", answer)
    return answer

# 启动 Flask 应用程序
if __name__ == "__main__":
    app.config['RESTFUL_JSON'] = {'ensure_ascii': False}
    app.run(host='0.0.0.0', port=5050)