# app.py

from flask import Flask, request, jsonify, render_template
import os
from dotenv import dotenv_values
from chatbot.fetch_data import fetch_initial_data
from chatbot.chat_logic import make_chain
import markdown2  # Ensure markdown2 is imported

# Load environment variables
config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Fetch initial data and create chain
chain = fetch_initial_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def get_answer():
    try:
        data = request.json
        question = data.get('question', '')
        chat_history = data.get('chat_history', '')
        organization_name = data.get('organization_name', '')
        organization_info = data.get('organization_info', '')
        contact_info = data.get('contact_info', '')

        # Create response using the chain
        response = chain({
            "question": question,
            "chat_history": chat_history,
            "organization_name": organization_name,
            "contact_info": contact_info,
            "organization_info": organization_info
        })

        answer = response['answer']
        chat_history = response['chat_history']

        # Append relevant document URL to the answer if available
        if 'source_documents' in response and response['source_documents']:
            most_relevant_document = response['source_documents'][0]
            most_relevant_article_url = most_relevant_document.metadata.get('url', 'No URL found')

            if most_relevant_article_url != 'No URL found':
                answer += f"\n\nFor more details, check out this article: [{most_relevant_document.metadata['title']}]({most_relevant_article_url})"

        # Format bot message using markdown
        bot_message = markdown2.markdown(answer, extras=["fenced-code-blocks", "code-friendly", "cuddled-lists"])

        return jsonify({"answer": bot_message})

    except Exception as e:
        # Handle errors gracefully
        error_message = f"Failed to get response. Error: {str(e)}"
        return jsonify({"answer": error_message})

if __name__ == '__main__':
    app.run(debug=True)
