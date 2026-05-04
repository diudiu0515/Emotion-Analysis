import os
from openai import OpenAI

def generate_data(num_samples=40, output_path="Dataset/robust_test.txt"):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None:
        raise ValueError("Please set DEEPSEEK_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    prompt = f"""
请生成{num_samples}条中文电影评论情感分类数据。

要求：
1. 二分类，1表示正向，0表示负向
2. 每行格式：标签\\t已分词句子
3. 句子中的词用空格分隔
4. 必须包含以下类型：
   - 简单正向句
   - 简单负向句
   - 否定句（如：不是很好）
   - 转折句（如：虽然...但是...）
   - 口语表达
5. 不要解释，只输出数据

示例：
1\t这 部 电影 虽然 节奏 慢 但是 结尾 很 感人
0\t演员 演技 不错 但是 剧情 实在 太 烂
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )

    text = response.choices[0].message.content.strip()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"已生成数据: {output_path}")


if __name__ == "__main__":
    generate_data()