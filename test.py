import re
import weaviate
import numpy as np
import string
import numpy as np
import re
import json

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

# 腾讯
from gensim.models import KeyedVectors
file = 'word/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)

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

# Initialize counter and interval for displaying progress


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
    
    # Calculate and display progress
    counter += 1
    if counter % interval == 0:
        print(f'Imported {counter} segments...')


# Read your data (assuming you have a way to get 'title', 'author', and 'segments')
for path in paths:
    title, author, segments = process_data(path)
    # Configure batch once
    client.batch.configure(batch_size=100)

    # Use the batch
    with client.batch as batch:
        for segment in segments:  # assuming 'segments' is a list of text segments from your article
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