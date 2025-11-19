# Используем мультистейдж сборку: сначала CUDA runtime, потом Python
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS cuda-base

FROM python:3.12-slim

# Копируем CUDA из первого образа
COPY --from=cuda-base /usr/local/cuda-12.6 /usr/local/cuda-12.6
COPY --from=cuda-base /usr/lib/x86_64-linux-gnu/libcuda* /usr/lib/x86_64-linux-gnu/
COPY --from=cuda-base /usr/lib/x86_64-linux-gnu/libnvidia* /usr/lib/x86_64-linux-gnu/

# Устанавливаем переменные окружения для CUDA
ENV PATH=/usr/local/cuda-12.6/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda-12.6

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем скрипт установки
COPY install_packages.sh ./
RUN chmod +x install_packages.sh

# Запускаем установку пакетов
RUN ./install_packages.sh

# Копируем остальные файлы
COPY app.py file_loader.py imp1.py models.py rag_methods.py docs .env ./

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
