import openai
from pinecone import Pinecone
import os
from tqdm import tqdm
import pandas as pd

# Define the file path
file_path = 'src/zach_content.csv'
# Read the CSV file into a DataFrame
content_df = pd.read_csv(file_path).to_dict(orient='records')
# Set up OpenAI and Pinecone API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(
    api_key=pinecone_api_key
)
# Create a Pinecone index if it doesn't exist
index_name = 'example-index'
# Connect to the Pinecone index
index = pc.Index(index_name)

def get_openai_embeddings(text, model="text-embedding-ada-002"):
    embeddings = openai.embeddings.create(input=text, model=model)
    return embeddings.data[0].embedding

for text in tqdm(content_df, desc="Processing text chunks"):
    if text['content'] and text['link']:
        try:
            embedding = get_openai_embeddings(text['content'])
            link = text['link']
            if text['source'] == 'Twitter':
                link = 'https://twitter.com/EcZachly/status/' + text['link']
            metadata = {
                'content': text['content'],
                'source': text['source']
            }
            index.upsert([(link, embedding, metadata)])
        except Exception as e:
            print(e)