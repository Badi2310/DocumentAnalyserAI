import os
import dotenv
import streamlit as st

from dotenv import load_dotenv
try:
    import langchain_core
    print("langchain_core:", langchain_core.__version__)
except ImportError:
    print("langchain_core не установлен")

import langchain_mistralai
print("langchain_mistralai: установлен и импортирован успешно")

try:
    import langchain
    print("langchain:", langchain.__version__)
except ImportError:
    print("langchain не установлен")

# Проверка chains (теперь в langchain_community)
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    print("create_history_aware_retriever и create_retrieval_chain: импортированы успешно из langchain.chains")
except ImportError as e:
    print(f"Импорт не удался: {e}. Установите langchain-community или проверьте версии.")

# Дополнительно: проверьте версию community
try:
    import langchain_community
    print("langchain-community:", langchain_community.__version__)
except ImportError:
    print("langchain-community не установлен")

# Тест других импортов
try:
    from langchain_mistralai import ChatMistralAI
    print("ChatMistralAI: импортирован успешно")
except ImportError:
    print("ChatMistralAI не найден")

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

dotenv.load_dotenv()

from mistralai import Mistral
# Access the secret
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("❌ MISTRAL_API_KEY not found in Colab secrets!")
else:
    print("✅ API key loaded successfully from Colab secrets!")

# Initialize Mistral client
client_1 = Mistral(api_key=api_key)

# Test connection
def test_connection():
    try:
        models = client_1.models.list()
        print("✅ Connected successfully!")
        print(f"Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 If key is not active yet, wait a few minutes and try again")

test_connection()


