# Introduction

This is a demo project, using multiple methods to do RAG, the default implementation is ALI. It can serve as an HTTP server.
Launch method:
1 export DASHSCOPE_API_KEY="xxx"
2 cd src
3 gunicorn main:app --workers 4 --bind 0.0.0.0:5050

Client command line:
curl localhost:5050/query -H 'Content-Type: application/json; charset=utf-8' -H 'Accept-Charset: utf-8' -d '{"query":"吴相铮是谁"}'
