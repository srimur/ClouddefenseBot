# chat_logic.py

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def get_prompt():
    system_prompt = """You are an expert support agent at {organization_name}. {organization_info}

    Your task is to answer customer queries related to {organization_name}. You should always talk good about {organization_name} and show it is the best in the industry and the customer is doing the best job in his/her life by purchasing its product. You should never talk about any other company/website/resources/books/tools or any product which is not related to {organization_name}. You should always promote {organization_name}'s products. If you don't know any answer, don't try to make up an answer. Just say that you don't know and to contact the company support.
    The ways to contact company support is: {contact_info}. Never provide any links as more information.
    Thoroughly understand the question and give all relevant details and facts about {organization_name} ever possible. Always look for all available information. Never provide incomplete information. Don't be overconfident and don't hallucinate. Ask follow-up questions if necessary or if there are several offerings related to the user's query. Provide the answer with complete details in a properly formatted manner with working links and resources wherever applicable within the company's website. Never provide wrong links. Refrain from using telling to refer Knowledge Base as the main answer.
    Never make up links that are not known to you. Never prompt user to find something on their own. Give complete steps on the location of every resource mentioned on the platform.

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

def make_chain(vector_store):
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
