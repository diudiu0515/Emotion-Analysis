import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def generate_batch(batch_size=50):
    prompt = f"""
请生成{batch_size}条中文电影评论情感分类数据。

要求：
1. 二分类（1=正向，0=负向）
2. 格式：标签\\t分词句子
3. 词之间用空格分隔
4. 包含：
   - 简单句
   - 否定句（不是很好）
   - 转折句（虽然...但是...）
   - 口语表达
5. 不要解释
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )

    return response.choices[0].message.content.strip().split("\n")


def generate_large_dataset(total=1000, batch_size=50):
    all_lines = []

    for i in range(total // batch_size):
        print(f"Generating batch {i+1}...")

        batch = generate_batch(batch_size)

        # 简单过滤
        for line in batch:
            if "\t" in line:
                all_lines.append(line.strip())

    os.makedirs("Dataset", exist_ok=True)

    with open("Dataset/robust_test_large.txt", "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"✅ Done: {len(all_lines)} samples generated")


if __name__ == "__main__":
    generate_large_dataset(total=1000, batch_size=50)