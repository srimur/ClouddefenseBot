# fetch_data.py

import requests
from bs4 import BeautifulSoup
import concurrent.futures
from langchain.text_splitter import MarkdownTextSplitter
from langchain.docstore.document import Document

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

def fetch_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_title = soup.find('h1').get_text(strip=True)  # Assuming titles are in <h1> tags
    article_content = soup.get_text(separator=' ')
    return article_title, article_content.strip()
