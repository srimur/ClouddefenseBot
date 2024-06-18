# process_data.py

import re
from langchain.text_splitter import MarkdownTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

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

def get_chroma_client():
    """
    Returns a chroma vector store instance.
    """
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="website_data",
        embedding_function=embedding_function,
        persist_directory="data/chroma"
    )
