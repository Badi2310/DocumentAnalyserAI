import os
import dotenv
import weaviate
import langchain
import streamlit as st

from weaviate.auth import AuthApiKey

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate

dotenv.load_dotenv()


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
    print("Документ успешно загружен в vector_db")
    return vector_db


def clear_weaviate_data():
    """Очистка всех данных из Weaviate Cloud"""
    try:
        WEAVIATE_CLUSTER = os.getenv("WEAVIATE_CLUSTER")
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")


        if not WEAVIATE_CLUSTER or not WEAVIATE_API_KEY:
            print("❌ Отсутствуют переменные окружения для Weaviate")
            st.toast("❌ Отсутствуют настройки подключения к БД", icon="⚠️")
            return


        WEAVIATE_URL = "https://" + WEAVIATE_CLUSTER


        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
        )


        schema = client.schema.get()


        if 'classes' in schema and len(schema['classes']) > 0:
            for class_obj in schema['classes']:
                class_name = class_obj['class']
                print(f"Удаление класса: {class_name}")
                client.schema.delete_class(class_name)
            print("✅ Weaviate БД полностью очищена")
            st.toast("✅ База данных очищена", icon="✅")
        else:
            print("ℹ️ БД уже пуста")
            st.toast("ℹ️ База данных уже пуста", icon="ℹ️")

    except Exception as e:
        print(f"❌ Ошибка очистки Weaviate: {e}")
        st.toast(f"❌ Ошибка очистки БД: {e}", icon="⚠️")

