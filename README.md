# CloudDefenseBot

This repository contains a Flask application that serves as a chatbot for CloudDefense.ai. The chatbot fetches support articles from the CloudDefense support website and uses OpenAI's GPT model to provide relevant answers to user queries.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- 

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.7 or higher
- `pip` (Python package installer)
- An OpenAI API key

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/srimur/ClouddefenseBot.git
   cd ClouddefenseBot
   ```
2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

## Running the Application

1. **Start the Flask application:**
   ```bash
   python app.py
   ```
2. **Open your web browser and go to:**
   ```
   http://127.0.0.1:5000 # Or http://localhost:5000/
   ```

## Usage

### Interact with the Chatbot:
- Click on the chatbot icon at the bottom right of the web page to open the chat interface.
- Type your query and press the "Send" button.
- The chatbot will respond with relevant information based on the CloudDefense support articles.


## Project Structure

```
ClouddefenseBot/
│
├── app.py                       # Main Flask application
├── templates/
│   └── index.html               # HTML template for the web interface
├── static/
│   └── chatbot-icon.png         # Chatbot icon
├── requirements.txt             # Python dependencies
└── .env                         # Environment variables file (not included in the repository)
```



