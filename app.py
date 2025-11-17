import locale
locale.getpreferredencoding = lambda: "UTF-8"

import streamlit as st
import os
import dotenv
import uuid

from dotenv import load_dotenv
from mistralai import Mistral
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

from imp1 import (
    load_pdf_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

MODELS = ["mistral-large-latest"]

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Language Chain MistralAI",
    page_icon="📑",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Заголовок страницы
st.markdown("""<h2 style="text-align: center;">📑🔍 <i>Проект LLM RAG</i></h2>""", unsafe_allow_html=True)

# Инициализация состояния сессии
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Уникальный ID сессии

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []  # Источники для RAG

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Привет! Чем могу помочь?"}
    ]  # Начальное сообщение ассистента

# Функция очистки чата
def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Привет! Чем могу помочь?"}]

st.button("Очистить чат", on_click=clear_chat, type="primary")

# Обработчик загрузки файлов
uploaded_files = st.file_uploader(
    "Загрузите документы",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True,
    key="rag_docs",
)

if uploaded_files:
    st.session_state.rag_sources.extend(uploaded_files)
    for file in uploaded_files:
        if file.type == "application/pdf":
            load_pdf_to_db(file)
    st.success(f"Загружено {len(uploaded_files)} документов.")

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Инициализация модели для потокового вывода
llm_stream = ChatMistralAI(
    model_name="mistral-large-latest",
    temperature=0.3,
    streaming=True
)

# Обработка нового ввода пользователя через st.chat_input
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_sources' not in st.session_state:
    st.session_state.rag_sources = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


if prompt := st.chat_input("Ваше сообщение"):
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Создаём placeholder для плавного обновления вывода
        placeholder = st.empty()
        
        # Формируем список сообщений для модели
        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]
        

        # Проверяем, есть ли загруженные документы
        if st.session_state.rag_sources:  # Если документы загружены, используем RAG
            response_stream = stream_llm_rag_response(messages)
        else:  # Иначе обычный LLM
            response_stream = stream_llm_response(llm_stream, messages)

        full_response = ""
        for chunk in response_stream:
            full_response += str(chunk)
            # Обновляем placeholder с накопленным текстом (плавный вывод)
            placeholder.markdown(full_response, unsafe_allow_html=True)

        # Добавляем финальный ответ в историю сообщений
        st.session_state.messages.append({"role": "assistant", "content": full_response})

