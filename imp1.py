import dotenv
import streamlit as st
import os
from dotenv import load_dotenv

# Импорты с проверкой
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

# Импорты для метрик
try:
    from sentence_transformers import SentenceTransformer, util
    print("sentence-transformers: импортирован успешно для метрик")
    METRICS_AVAILABLE = True
except ImportError:
    print("sentence-transformers не установлен - метрики будут недоступны")
    METRICS_AVAILABLE = False

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from file_loader import PDFProcessor

dotenv.load_dotenv()

print("weaviate version:", weaviate.__version__)
print("langchain version:", langchain.__version__)

# Константы
DB_DOCS_LIMIT = 10

# Инициализация модели
api_key = os.getenv("MISTRAL_API_KEY")
model = ChatMistralAI(
    api_key=api_key,
    model="mistral-large-latest",
    temperature=0.7,
    max_tokens=1024  # Увеличил для более полных ответов
)
output_parser = StrOutputParser()

# Инициализация модели для метрик
if METRICS_AVAILABLE:
    metrics_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("✅ Модель для метрик загружена")


def calculate_answer_relevance(question, answer, context):
    """Вычисление релевантности ответа к вопросу и контексту"""
    if not METRICS_AVAILABLE:
        return None
    
    try:
        # Эмбеддинги
        question_emb = metrics_model.encode(question, convert_to_tensor=True)
        answer_emb = metrics_model.encode(answer, convert_to_tensor=True)
        context_emb = metrics_model.encode(context, convert_to_tensor=True)
        
        # Косинусное сходство
        q_a_similarity = util.pytorch_cos_sim(question_emb, answer_emb).item()
        a_c_similarity = util.pytorch_cos_sim(answer_emb, context_emb).item()
        
        print(f"\n📊 МЕТРИКИ РЕЛЕВАНТНОСТИ:")
        print(f"  Вопрос ↔ Ответ: {q_a_similarity:.4f}")
        print(f"  Ответ ↔ Контекст: {a_c_similarity:.4f}")
        print(f"  Средняя релевантность: {(q_a_similarity + a_c_similarity) / 2:.4f}")
        
        return {
            "question_answer_similarity": q_a_similarity,
            "answer_context_similarity": a_c_similarity,
            "average_relevance": (q_a_similarity + a_c_similarity) / 2
        }
    except Exception as e:
        print(f"❌ Ошибка вычисления метрик: {e}")
        return None


def calculate_context_precision(retrieved_docs, user_query):
    """Вычисление точности извлечения контекста"""
    if not METRICS_AVAILABLE or not retrieved_docs:
        return None
    
    try:
        query_emb = metrics_model.encode(user_query, convert_to_tensor=True)
        
        scores = []
        for doc in retrieved_docs:
            doc_emb = metrics_model.encode(doc.page_content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_emb, doc_emb).item()
            scores.append(similarity)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"\n📊 МЕТРИКИ КОНТЕКСТА:")
        print(f"  Извлечено документов: {len(retrieved_docs)}")
        print(f"  Средняя релевантность контекста: {avg_score:.4f}")
        print(f"  Релевантность по документам: {[f'{s:.3f}' for s in scores]}")
        
        return {
            "num_retrieved": len(retrieved_docs),
            "average_context_score": avg_score,
            "individual_scores": scores
        }
    except Exception as e:
        print(f"❌ Ошибка вычисления precision: {e}")
        return None


def load_pdf_to_db(pdf_file):
    """Загрузка PDF файла в векторную базу данных"""
    docs = []
    os.makedirs("docs", exist_ok=True)

    # ИСПРАВЛЕНО: Проверяем количество уникальных имен файлов
    unique_sources = list(set(st.session_state.rag_sources))
    print(f"📚 Текущее количество уникальных документов: {len(unique_sources)}/{DB_DOCS_LIMIT}")

    if pdf_file.name not in st.session_state.rag_sources:
        if len(unique_sources) < DB_DOCS_LIMIT:
            file_path = os.path.join("docs", pdf_file.name)
            with open(file_path, "wb") as f:
                f.write(pdf_file.read())

            try:
                loader = PDFProcessor()
                data = loader.process(file_path)

                for page_num, page_text in enumerate(data):
                    doc = Document(
                        page_content=page_text,
                        metadata={"source": pdf_file.name, "page": page_num + 1},
                    )
                    docs.append(doc)
                
                print(f"Извлечено {len(docs)} страниц из {pdf_file.name}")

                # Добавляем имя файла в список загруженных
                st.session_state.rag_sources.append(pdf_file.name)
                
                # Разбиваем на чанки и загружаем в БД
                _split_and_load_docs(docs)
                
                st.toast(f"✅ {pdf_file.name} успешно загружен", icon="✅")

            except Exception as e:
                st.toast(f"Ошибка загрузки PDF {pdf_file.name}: {e}", icon="⚠️")
                print(f"Ошибка загрузки PDF {pdf_file.name}: {e}")
        else:
            st.error(f"Достигнут лимит уникальных документов ({DB_DOCS_LIMIT}).")
            print(f"❌ Лимит документов достигнут: {len(unique_sources)}/{DB_DOCS_LIMIT}")
    else:
        st.warning(f"Документ {pdf_file.name} уже загружен.")
        print(f"⚠️ Документ {pdf_file.name} уже в базе")


def initialize_vector_db(docs):
    """Инициализация векторной базы данных Weaviate"""
    WEAVIATE_CLUSTER = os.getenv("WEAVIATE_CLUSTER")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    if not WEAVIATE_CLUSTER or not WEAVIATE_API_KEY:
        raise ValueError("Отсутствуют переменные окружения WEAVIATE_CLUSTER или WEAVIATE_API_KEY")
    
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
    print(f"✅ Документы успешно загружены в vector_db ({len(docs)} чанков)")
    return vector_db


def _split_and_load_docs(pages):
    """Разделение текста на чанки и загрузка в векторную БД"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(pages)
    
    print(f"Создано {len(docs)} чанков из {len(pages)} страниц")

    if "vector_db" not in st.session_state or st.session_state.vector_db is None:
        st.session_state.vector_db = initialize_vector_db(docs)
        print(f"Инициализирована новая база с {len(docs)} документами")
    else:
        st.session_state.vector_db.add_documents(docs)
        print(f"Добавлено {len(docs)} документов в существующую базу")


def stream_llm_response(llm, messages):
    """Стриминг ответа модели БЕЗ RAG"""
    response_message = ""
    for chunk in llm.stream(messages):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            response_message += chunk.content
            yield chunk.content
        else:
            continue
    
    # Добавляем полный ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": response_message})


def _get_context_retriever_chain(vector_db, model):
    """Создание retriever chain для поиска контекста"""
    if vector_db is None:
        print("⚠️ ВНИМАНИЕ: vector_db = None, инициализируем пустую базу")
        st.session_state.vector_db = initialize_vector_db([])
        vector_db = st.session_state.vector_db

    retriever = vector_db.as_retriever(
        search_type="mmr",  
        search_kwargs={
            "k": 15,  
            "fetch_k": 30,  
            "lambda_mult": 0.7  
        }
    )
    
    test_results = retriever.get_relevant_documents("test")
    print(f"Retriever тест: найдено {len(test_results)} документов")

    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"),
        ("user", """Generate a comprehensive search query to find ALL key themes and topics from the document. 
        When asked about document contents, search for: main topics, key people, numbers, events, and overall structure."""),
    ])

    retriever_chain = create_history_aware_retriever(model, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(model):
    """Создание RAG цепочки с историей разговора"""
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, model)

    if st.session_state.rag_sources:
        unique_sources = list(set(
            item.name if hasattr(item, 'name') else str(item) 
            for item in st.session_state.rag_sources
        ))
        loaded_docs_list = ", ".join(unique_sources)
    else:
        loaded_docs_list = "Нет загруженных документов"

    print(f"=== Loaded documents: {loaded_docs_list} ===")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        f"""You are a helpful AI assistant. Answer user questions based on the provided context.

    CONTEXT FROM DOCUMENTS:
    {{context}}

    LOADED DOCUMENTS: {loaded_docs_list}

    INSTRUCTIONS:
    - The context above contains the actual content from the loaded PDF documents
    - When asked "what's inside the file" or similar questions, the context IS the file content
    - Answer directly based on what you see in the context
    - If asked about file contents, summarize or quote the context directly
    - The context shows exactly what is written in the documents
    - If the context is empty or doesn't contain relevant info, say so clearly
    - If you see [Image: ...] that's a describe of image in text. Use it in answer if you need."""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(model, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



def stream_llm_rag_response(messages):
    """Стриминг ответа модели С RAG + метрики"""
    model = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-large-latest",
        temperature=0.3,
        max_tokens=16248,
        streaming=True
    )
    
    user_query = messages[-1].content
    print(f"\n{'='*60}")
    print(f"USER QUERY: {user_query}")
    print(f"{'='*60}")
    
    if st.session_state.vector_db is None:
        print("❌ ОШИБКА: vector_db не инициализирована!")
        error_msg = "⚠️ База данных не инициализирована. Загрузите документ."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        yield error_msg
        return
    
    retrieved_docs = []
    full_context = ""
    try:
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(user_query)
        print(f"\n📚 RETRIEVED: {len(retrieved_docs)} документов")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"\nДокумент {i+1}:")
            print(f"  Источник: {doc.metadata.get('source')}")
            print(f"  Страница: {doc.metadata.get('page')}")
            print(f"  Текст: {doc.page_content[:200]}...")
        
        # Полный контекст
        full_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"\n📄 ПОЛНЫЙ КОНТЕКСТ ({len(full_context)} символов):")
        print(full_context[:500] + "...\n")
        
        # МЕТРИКИ КОНТЕКСТА
        calculate_context_precision(retrieved_docs, user_query)
        
    except Exception as e:
        print(f"❌ Ошибка при retrieval: {e}")
    
    # Создание RAG-цепочки и стриминг
    conversation_rag_chain = get_conversational_rag_chain(model)
    response_message = "*(RAG Response)*\n"
    
    for chunk in conversation_rag_chain.pick("answer").stream({
        "messages": messages[:-1], 
        "input": user_query
    }):
        response_message += chunk
        yield chunk
    
    # МЕТРИКИ ОТВЕТА
    if full_context and len(response_message) > 20:
        calculate_answer_relevance(user_query, response_message, full_context)
    
    # Добавление в сессию
    st.session_state.messages.append({"role": "assistant", "content": response_message})
    
    print(f"\n{'='*60}")
    print("✅ ОТВЕТ ЗАВЕРШЕН")
    print(f"  Длина ответа: {len(response_message)} символов")
    print(f"{'='*60}\n")


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
