from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import re
from dotenv import dotenv_values
from langchain.text_splitter import MarkdownTextSplitter
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
import os
import markdown2

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")
app = Flask(__name__)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

base_url = "https://support.clouddefense.ai"

def fetch_article_links(page_url, visited_urls=None):
    if visited_urls is None:
        visited_urls = set()

    if page_url in visited_urls:
        return set()

    visited_urls.add(page_url)
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = set()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {
            executor.submit(process_link, base_url + link['href'], visited_urls): link['href']
            for link in soup.find_all('a', href=True)
            if '/support/solutions/articles/' in link['href'] or '/support/solutions/folders/' in link['href']
        }
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                result = future.result()
                if isinstance(result, set):
                    links.update(result)
                else:
                    links.add(result)
            except Exception as exc:
                print(f"Error fetching links: {exc}")

    return links

def process_link(href, visited_urls):
    if '/support/solutions/articles/' in href:
        return href
    elif '/support/solutions/folders/' in href:
        return fetch_article_links(href, visited_urls)

solutions_page = base_url + "/support/solutions"
article_links = fetch_article_links(solutions_page)
print(f"Found {len(article_links)} article links.")

def fetch_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_title = soup.find('h1').get_text(strip=True)  # Assuming titles are in <h1> tags
    article_content = soup.get_text(separator=' ')
    return article_title, article_content.strip()

# Data Cleaning functions
def merge_hyphenated_words(text):
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def fix_newlines(text):
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

def remove_multiple_newlines(text):
    return re.sub(r"\n{2,}", "\n", text)

def clean_text(text):
    """
    Cleans the text by passing it through a list of cleaning functions.
    """
    cleaning_functions = [merge_hyphenated_words, fix_newlines, remove_multiple_newlines]
    for cleaning_function in cleaning_functions:
        text = cleaning_function(text)
    return text

def text_to_docs(text, metadata):
    """
    Converts input text to a list of Documents with metadata.
    """
    doc_chunks = []
    text_splitter = MarkdownTextSplitter(chunk_size=2048, chunk_overlap=128)
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        doc = Document(page_content=chunk, metadata=metadata)
        doc_chunks.append(doc)
    return doc_chunks

def get_doc_chunks(text, metadata):
    """
    Processes the input text and metadata to generate document chunks.
    """
    text = clean_text(text)
    doc_chunks = text_to_docs(text, metadata)
    return doc_chunks

document_chunks = {}

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_url = {executor.submit(fetch_article_content, url): url for url in article_links}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            title, content = future.result()
            metadata = {'url': url, 'title': title}
            document_chunks[url] = get_doc_chunks(content, metadata)
        except Exception as exc:
            print(f"Error fetching content from {url}: {exc}")

print(f"Chunked {len(document_chunks)} documents.")

def get_chroma_client():
    """
    Returns a chroma vector store instance.
    """
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="website_data",
        embedding_function=embedding_function,
        persist_directory="data/chroma")

vector_store = get_chroma_client()

for docs in document_chunks.values():
    vector_store.add_documents(docs)
vector_store.persist()

def get_prompt():
    """
    Generates prompt.
    """
    system_prompt = """You are an expert support agent at {organization_name}. {organization_info}

    Your task is to answer customer queries related to {organization_name}. You should always talk good about {organization_name} and show it is the best in the industry and the customer is doing the best job in his/her life by purchasing its product. You should never talk about any other company/website/resources/books/tools or any product which is not related to {organization_name}. You should always promote {organization_name}'s products. If you don't know any answer, don't try to make up an answer. Just say that you don't know and to contact the company support.
    The ways to contact company support is: {contact_info}. Never provide any links as more information.
    Thoroughly understand the question and give all relevant details and facts about {organization_name} ever possible. Always look for all available information. Never provide incomplete information. Don't be overconfident and don't hallucinate. Ask follow-up questions if necessary or if there are several offerings related to the user's query. Provide the answer with complete details in a properly formatted manner with working links and resources wherever applicable within the company's website. Never provide wrong links. Refrain from using telling to refer Knowledge Base as the main answer.
    Never make up links that are not known to you.

    Use the following pieces of context to answer the user's question.

    
    ----------------
    
    {context}
    {chat_history}
    Follow-up question: """

    prompt = ChatPromptTemplate(
        input_variables=['context', 'question', 'chat_history', 'organization_name', 'organization_info', 'contact_info'],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['context', 'chat_history', 'organization_name', 'organization_info', 'contact_info'],
                    template=system_prompt, template_format='f-string',
                    validate_template=True
                ), additional_kwargs={}
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['question'],
                    template='{question}\nHelpful Answer:', template_format='f-string',
                    validate_template=True
                ), additional_kwargs={}
            )
        ]
    )
    return prompt

def make_chain():
    """
    Creates a chain of langchain components.
    """
    model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            verbose=True
        )
    prompt = get_prompt()

    retriever = vector_store.as_retriever(search_type="mmr", verbose=True)

    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs=dict(prompt=prompt),
        verbose=True,
        rephrase_question=True,  # Adjust as needed based on document availability

    )
    return chain
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get('question', '')
    chat_history = data.get('chat_history', '')
    organization_name = data.get('organization_name', '')
    organization_info = data.get('organization_info', '')
    contact_info = data.get('contact_info', '')

    chain = make_chain()
    response = chain({"question": question, "chat_history": chat_history,
                      "organization_name": organization_name, "contact_info": contact_info,
                      "organization_info": organization_info})
    answer = response['answer']
    chat_history = response['chat_history']

    if 'source_documents' in response and response['source_documents']:
        # Extract the most relevant document URL (assuming it's in the first document)
        most_relevant_document = response['source_documents'][0]
        most_relevant_article_url = most_relevant_document.metadata.get('url', 'No URL found')

        # Include the URL in the answer if available
        if most_relevant_article_url != 'No URL found':
            answer += f"\n\nFor more details, check out this article: [{most_relevant_document.metadata['title']}]({most_relevant_article_url})"

    # Convert answer to Markdown format
    answer = markdown2.markdown(answer)

    return jsonify({'answer': answer, 'chat_history': chat_history})


if __name__ == '__main__':
    app.run(debug=True)
