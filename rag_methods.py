import os
import re
import uuid

import dotenv
from dotenv import load_dotenv

import streamlit as st
from IPython.display import display, Markdown

import weaviate
from weaviate.auth import AuthApiKey

from mistralai import Mistral
from langchain_mistralai import ChatMistralAI

import langchain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Weaviate

dotenv.load_dotenv()


print(weaviate.__version__)
print(langchain.__version__)

DB_DOCS_LIMIT = 10


import os

def load_pdf_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        os.makedirs("docs", exist_ok=True)
        
        for doc_file in st.session_state.rag_docs:
            if doc_file.name.endswith(".pdf") and doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    file_path = os.path.join("docs", doc_file.name)
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())
                    
                    try:
                        loader = PyPDFLoader(file_path)
                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                    except Exception as e:
                        st.toast(f"Error loading PDF {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading PDF {doc_file.name}: {e}")
                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")
        
        if docs:
            _split_and_load_docs(docs)
            loaded_names = ", ".join([doc_file.name for doc_file in st.session_state.rag_docs if doc_file.name.endswith(".pdf")])
            st.toast(f"Documents {loaded_names} loaded successfully.", icon="✅")


#создание векторного хранилща
def initialize_vector_db(docs):
    WEAVIATE_CLUSTER = os.getenv("WEAVIATE_CLUSTER")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

    WEAVIATE_URL = "https://" + WEAVIATE_CLUSTER
    auth = AuthApiKey(api_key=WEAVIATE_API_KEY)
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=auth
        )
    
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vector_db = Weaviate.from_documents(
        docs, embeddings, client=client, by_text=False
        )
    
    return vector_db


# Разделение текста на чанки (chunks)
def _split_and_load_docs(pages):
    # Создание экземпляра разделителя текста
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Размер каждого чанка в символах
        chunk_overlap=20   # Перекрытие между чанками в символах
    )
    # Применение разделителя к документам (pages)
    docs = text_splitter.split_documents(pages)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(pages)
    else:
        st.session_state.vector_db.add_documents(docs)



def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

api_key = os.getenv("MISTRAL_API_KEY")
model = ChatMistralAI(
    api_key=api_key,
    model="mistral-large-latest",
    temperature=0.7,
    max_tokens=512
)
output_parser = StrOutputParser()

def _get_context_retriever_chain(vector_db, model):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{user}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])

    retriever_chain = create_history_aware_retriever(model, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(model):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, model)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant. You will have to answer to user's queries.
        You will have some context to help with your answers, but not always would be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(model, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})
