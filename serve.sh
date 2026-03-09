#!/bin/bash
# 启动 MkDocs 开发服务器
# 用法: ./serve.sh
# 然后浏览器打开 http://127.0.0.1:8000

conda run -n test mkdocs serve -a 127.0.0.1:8800
