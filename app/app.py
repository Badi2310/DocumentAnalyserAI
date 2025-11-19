import locale

import streamlit as st

from dotenv import load_dotenv

from utils.session_state import init_session_state, clear_chat
from utils.file_loader import handle_uploaded_files
from utils.database import clear_weaviate_data

from models.llm import init_llm
from models.rag import generate_response

locale.getpreferredencoding = lambda: "UTF-8"



init_session_state()

col1, col2 = st.columns(2)

with col1:
    st.button("Очистить чат", on_click=clear_chat, type="primary")

with col2:
    if st.button("🗑️ Очистить БД", type="secondary"):
        clear_weaviate_data()
        st.session_state.vector_db = None
        st.session_state.rag_sources = []
        st.session_state.uploader_key += 1
        st.rerun()



uploaded_files = st.file_uploader(
    "Загрузите документы",
    type=["pdf","txt","docx","md"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_files:
    count = handle_uploaded_files(uploaded_files, st.session_state.rag_sources)
    st.success(f"Загружено {count} документов.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



load_dotenv()

llm_stream = init_llm()



if prompt := st.chat_input("Ваше сообщение"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in generate_response(llm_stream, st.session_state.messages, st.session_state.rag_sources):
            full_response += str(chunk)
            placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant","content":full_response})
