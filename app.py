from flask import Flask, request, jsonify, render_template_string
import os
import openai
import json
from openai import OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

from pinecone import Pinecone
app = Flask(__name__)
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_db = Pinecone(
    api_key=pinecone_api_key
)

# Create a Pinecone index if it doesn't exist
index_name = 'example-index'
# Connect to the Pinecone index
index = pinecone_db.Index(index_name)
# Create a Pinecone index if it doesn't exist
index_name = 'example-index'

# HTML template for the form page
form_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Query Form</title>
</head>
<body>
    <h1>Submit Your Query</h1>
    <form id="queryForm" action="/query">
        <label for="query">Query:</label>
        <input type="text" id="query" name="query">
        <br/>
        <br/>
        <label for="top_k">Top N results:</label>
        <input type="number" id="top_k" name="top_k">
        <br/>
        <br/>
        <button type="submit">Submit</button>
    </form>
    <div id="result"></div>
    <script>
        document.getElementById('queryForm').addEventListener('submit', function(event) {
            event.preventDefault();
            var query = document.getElementById('query').value;
            var topK = document.getElementById('top_k').value;
            fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query, top_k: topK })
            })
            .then(response => {
                return response.json()
            })
            .then(data => {
                var result = document.getElementById('result')
                result.innerHTML = '';
                var llmResponse = document.createElement('p')
                llmResponse.textContent = data['response'];
                var contextualHeader = document.createElement("h1");
                contextualHeader.textContent = 'Augmented Results';
                result.appendChild(llmResponse);
                result.appendChild(contextualHeader);
                var newList = document.createElement("ul");
                newList.id = 'rag_data';
               
                data['matches'].forEach(function(value){ 
                    console.log(value);
                    var newListItem = document.createElement("li");
                    var newLink = document.createElement("a");
                    newLink.href = value['id']
                    newLink.textContent = value['metadata']['source']
                    var newParagraph = document.createElement("p");
                    newParagraph.textContent = value['metadata']['content']
                    newListItem.appendChild(newLink);
                    newListItem.appendChild(newParagraph);
                    newList.appendChild(newListItem);
                })
                result.appendChild(newList);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(form_page)


def get_feedback(user_prompt: str) -> str:
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": """
          You are a staff data engineer trying to teach people! 
          If you use any augmented information please provide the link!
          """
           },
          {"role": "user", "content": user_prompt},
      ],
      temperature=0.2
    )
    return response.choices[0].message.content


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    openai_embeddings = openai.embeddings.create(input=data['query'], model="text-embedding-ada-002")
    vector = openai_embeddings.data[0].embedding
    try:
        top_k = int(data['top_k'])
    except Exception as e:
        top_k = 0

    matches = []
    if top_k > 0:
        result = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        matches = list(map(lambda x: {
            'id': x['id'],
            'metadata': x['metadata']
        }, result['matches']))

    jsonified_matches = '------- here is some augmented info: ' + str(json.dumps(matches)) if len(matches) > 0 else ''
    prompt = str(data['query']) + str(jsonified_matches)
    print(prompt)
    response = get_feedback(prompt)
    print(response)
    return {'matches': matches, 'response': response}



if __name__ == '__main__':
    app.run(debug=True)
