import dotenv
import streamlit as st
import time


from dotenv import load_dotenv
try:
    import weaviate
    print("weaviate: установлен и импортирован успешно")
except ImportError:
    print("weaviate не установлен")

try:
    from weaviate.auth import AuthApiKey
    print("AuthApiKey из weaviate.auth: импортирован успешно")
except ImportError:
    print("AuthApiKey из weaviate.auth не найден")

try:
    import mistralai
    print("mistralai: установлен и импортирован успешно")
except ImportError:
    print("mistralai не установлен")

try:
    from mistralai import Mistral
    print("Mistral: импортирован успешно")
except ImportError:
    print("Mistral не найден")

try:
    import langchain_mistralai
    print("langchain_mistralai: установлен и импортирован успешно")
except ImportError:
    print("langchain_mistralai не установлен")

try:
    from langchain_mistralai import ChatMistralAI
    print("ChatMistralAI: импортирован успешно")
except ImportError:
    print("ChatMistralAI не найден")

try:
    import langchain
    print("langchain:", langchain.__version__)
except ImportError:
    print("langchain не установлен")

try:
    import langchain_core
    print("langchain_core:", langchain_core.__version__)
except ImportError:
    print("langchain_core не установлен")

try:
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    print("RunnablePassthrough, ChatPromptTemplate, MessagesPlaceholder, StrOutputParser из langchain_core импортированы успешно")
except ImportError as e:
    print(f"Импорт из langchain_core не удался: {e}")

try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    print("create_history_aware_retriever и create_retrieval_chain: импортированы успешно из langchain.chains")
except ImportError as e:
    print(f"Импорт create_history_aware_retriever и create_retrieval_chain не удался: {e}. Проверьте langchain-community и версии")

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    print("create_stuff_documents_chain: импортирован успешно")
except ImportError as e:
    print(f"Импорт create_stuff_documents_chain не удался: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("RecursiveCharacterTextSplitter: импортирован успешно")
except ImportError:
    print("langchain_text_splitters не установлен или RecursiveCharacterTextSplitter не найден")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings: импортирован успешно")
except ImportError:
    print("langchain_huggingface не установлен или HuggingFaceEmbeddings не найден")

try:
    from langchain_community.document_loaders import PyPDFLoader
    print("PyPDFLoader: импортирован успешно")
except ImportError:
    print("langchain_community.document_loaders не установлен или PyPDFLoader не найден")

try:
    from langchain_community.vectorstores import Weaviate
    print("Weaviate из langchain_community.vectorstores: импортирован успешно")
except ImportError:
    print("langchain_community.vectorstores не установлен или Weaviate не найден")

try:
    import langchain_community
    print("langchain-community:", langchain_community.__version__)
except ImportError:
    print("langchain-community не установлен")

from mistralai import Mistral

dotenv.load_dotenv()


print(weaviate.__version__)
print(langchain.__version__)

DB_DOCS_LIMIT = 10


import os

def load_pdf_to_db(pdf_file):
    docs = []
    os.makedirs("docs", exist_ok=True)
    if pdf_file.name not in st.session_state.rag_sources:
        if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
            file_path = os.path.join("docs", pdf_file.name)
            with open(file_path, "wb") as f:
                f.write(pdf_file.read())
            try:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
                st.session_state.rag_sources.append(pdf_file.name)
            except Exception as e:
                st.toast(f"Ошибка загрузки PDF {pdf_file.name}: {e}", icon="⚠️")
                print(f"Ошибка загрузки PDF {pdf_file.name}: {e}")
        else:
            st.error(f"Достигнут лимит документов ({DB_DOCS_LIMIT}).")
    
    if docs:
        _split_and_load_docs(docs)
        st.toast(f"Документ {pdf_file.name} успешно загружен.", icon="✅")


#создание векторного хранилща

def initialize_vector_db(docs):
    WEAVIATE_CLUSTER = os.getenv("WEAVIATE_CLUSTER")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_URL = "https://" + WEAVIATE_CLUSTER
    
    # Инициализация клиента для Weaviate 3.x
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
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



from langchain_core.messages import AIMessageChunk  # Для типизации, если нужно

def stream_llm_response(llm, messages):  # Переименовал llm_stream -> llm для ясности
    response_message = ""
    for chunk in llm.stream(messages):
        if isinstance(chunk, AIMessageChunk) and chunk.content:  # Проверяем, что это текст
            response_message += chunk.content
            yield chunk.content  # Yield только текст, без метаданных
        else:
            # Если чанк не текст (редко), пропускаем
            continue
    
    # Добавляем полный ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": response_message})

def stream_llm_rag_response(messages):  # Убрал llm_stream, используем модель напрямую
    time.sleep(5)
    model = ChatMistralAI(  # Или возьмите из глобальной
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-large-latest",
        temperature=0.7,
        max_tokens=512,
        streaming=True  # Включаем стриминг явно
    )
    
    conversation_rag_chain = get_conversational_rag_chain(model)
    response_message = "*(RAG Response)*"
    
    # Stream цепочки: используем .stream с input
    for chunk in conversation_rag_chain.pick("answer").stream(
        {"messages": messages[:-1], "input": messages[-1].content}
    ):
        if hasattr(chunk, 'content') and chunk.content and chunk.content.strip():  # Для совместимости
            text = chunk.content.replace('\n', ' ')
            response_message += text
            yield text  # Только текст
        else:
            text = str(chunk).replace('\n', ' ')
            response_message += text  # Fallback для не-AIMessage
            yield text
    
    # Добавляем полный в историю
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
