import re
import weaviate
import numpy as np
import string
import numpy as np
import re
import json
import os
import tarfile
from gensim.models import KeyedVectors

from flask import Flask, request, jsonify

app = Flask(__name__)

counter = 0
interval = 200

auth_config = weaviate.AuthApiKey(api_key="cSS9XqOYrQ47rPmzCogrKYYm6rDhaZ7DJdCX")

client = weaviate.Client(
    url="https://test-s3sbv82a.weaviate.network",
    auth_client_secret=auth_config
)
paths = ["test-dataset-1.md", "test-dataset-2.md"]

# 腾讯向量词
def extract_tar_gz(file_path, extract_path):
    # 解压缩文件
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

# 检查特定文件是否存在
specific_file_path = 'tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
tar_gz_path = 'tencent-ailab-embedding-zh-d100-v0.2.0-s.tar.gz'

if not os.path.exists(specific_file_path):
    print("Embedding file not found. Extracting now...")
    extract_tar_gz(tar_gz_path, '')
    print("Extraction complete.")

wv_from_text = KeyedVectors.load_word2vec_format(specific_file_path, binary=False)

# 清除类
client.schema.delete_class("Article")

# 创建类
article_class = {
    "class": "Article",
    "description": "A class to represent articles",
    "properties": [
        {
            "name": "title",
            "description": "The title of the article",
            "dataType": ["text"]
        },
        {
            "name": "author",
            "description": "The author of the article",
            "dataType": ["text"]
        },
        {
            "name": "content",
            "description": "The content of the article",
            "dataType": ["text"]
        }
    ]
}

# 在Weaviate中添加类定义
client.schema.create_class(article_class)



def process_data(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    title_pattern = r"# (.*?)\n"
    author_pattern = r">\s*(.*?)\n"
    content_pattern = r"(?:>\s*.*?\n)([\s\S]*)"
    title = re.search(title_pattern, content).group(1)
    author = re.search(author_pattern, content).group(1)
    article_content = re.search(content_pattern, content).group(1)
    punctuation = ' '
    article_content = re.sub(f"[{re.escape(punctuation)}]", "", article_content)
    article_content = article_content.replace("\n", "")
    pattern = r'\*\*(.*?)\*\*|。|；'
    segments = re.split(pattern, article_content)
    # 过滤空字符串和None
    segments = [seg.strip() for seg in segments if seg and seg.strip() not in ['##']]
    return title, author, segments

# Function to add articles to batch
def add_article_to_batch(batch, title, author, segment, wv_from_text):
    global counter
    sentence_vector = [wv_from_text[word] for word in segment if word in wv_from_text.key_to_index]
    if sentence_vector:
        sentence_vector_mean = np.mean(sentence_vector, axis=0).tolist()
    else:
        sentence_vector_mean = []
    properties = {
        "title": title,
        "author": author,
        "content": segment.strip()
    }
    batch.add_data_object(properties, 'Article', vector=sentence_vector_mean)
    
    counter += 1
    if counter % interval == 0:
        print(f'Imported {counter} segments...')


# 读取数据
for path in paths:
    title, author, segments = process_data(path)
    # Configure batch once
    client.batch.configure(batch_size=100)

    # 循环读取数据
    with client.batch as batch:
        for segment in segments:  
            add_article_to_batch(batch, title, author, segment, wv_from_text)

    print(f'Finished importing {counter} segments.')

@app.route('/search', methods=['GET'])
def search():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({"error": "Keyword is required"}), 400
        
    keyword_vector = [wv_from_text[word] for word in keyword if word in wv_from_text.key_to_index]
    keyword_vector_mean = np.mean(keyword_vector, axis=0).tolist()

    result = (
        client.query
        .get("Article", ["title", "content","author"])
        .with_hybrid(keyword, alpha=0.5,properties=["content"],vector=keyword_vector_mean)
        .with_limit(5)
        .do()
    )

    return json.dumps(result, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)