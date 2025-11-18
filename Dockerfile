FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Установите pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Симлинки
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

COPY app.py file_loader.py imp1.py models.py rag_methods.py install_packages.sh docs .env ./
RUN chmod +x install_packages.sh

RUN ./install_packages.sh

EXPOSE 8501

CMD ["python3.12", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
