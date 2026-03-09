#!/bin/bash
# 从所有 .dot 文件生成 SVG 知识图
# 用法: ./build-graphs.sh
# 需要: brew install graphviz

DOT_CMD="/opt/homebrew/bin/dot"

if ! command -v "$DOT_CMD" &> /dev/null; then
  DOT_CMD="dot"
  if ! command -v "$DOT_CMD" &> /dev/null; then
    echo "错误: 未找到 Graphviz (dot)，请运行 brew install graphviz"
    exit 1
  fi
fi

count=0
for dotfile in domains/*/knowledge-graph.dot; do
  if [ -f "$dotfile" ]; then
    dir=$(dirname "$dotfile")
    mkdir -p "$dir/images"
    outfile="$dir/images/knowledge-graph.svg"
    echo "生成: $dotfile → $outfile"
    "$DOT_CMD" -Tsvg "$dotfile" -o "$outfile"
    count=$((count + 1))
  fi
done

echo "✓ 共生成 $count 张知识图"
