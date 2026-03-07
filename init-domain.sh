#!/bin/bash
# 初始化新领域学习文件夹
# 用法: ./init-domain.sh <领域名称>
# 示例: ./init-domain.sh reinforcement-learning

if [ -z "$1" ]; then
    echo "用法: ./init-domain.sh <领域名称>"
    echo "示例: ./init-domain.sh reinforcement-learning"
    exit 1
fi

DOMAIN_NAME="$1"
DOMAIN_DIR="domains/${DOMAIN_NAME}"
TEMPLATE_DIR="domains/_template"

if [ -d "$DOMAIN_DIR" ]; then
    echo "错误: 领域 '${DOMAIN_NAME}' 已存在"
    exit 1
fi

if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "错误: 模板目录不存在"
    exit 1
fi

cp -r "$TEMPLATE_DIR" "$DOMAIN_DIR"

TODAY=$(date +%Y-%m-%d)
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/YYYY-MM-DD/${TODAY}/g" "$DOMAIN_DIR/README.md"
    sed -i '' "s/\[领域名称\]/${DOMAIN_NAME}/g" "$DOMAIN_DIR/README.md"
else
    sed -i "s/YYYY-MM-DD/${TODAY}/g" "$DOMAIN_DIR/README.md"
    sed -i "s/\[领域名称\]/${DOMAIN_NAME}/g" "$DOMAIN_DIR/README.md"
fi

echo "✓ 已创建领域学习目录: ${DOMAIN_DIR}"
echo ""
echo "目录结构:"
find "$DOMAIN_DIR" -type f | sort | while read -r f; do
    echo "  $f"
done
echo ""
echo "下一步: 告诉 Agent '我想学习 ${DOMAIN_NAME}' 来开始学习"
