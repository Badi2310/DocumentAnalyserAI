docker build -t document-analyser .
docker run --gpus all -p 8501:8501 document-analyser
