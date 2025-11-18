#!/bin/bash

# Bash script to install Python packages individually using pip with --force-reinstall
# This installs each package one by one, which can help with debugging or handling dependencies separately.
# If errors occur due to compatibility, you can add flags like --no-deps or --ignore-installed as needed.

# Update pip first (optional, but recommended)
python3.12 -m pip install --upgrade pip

# Install each package individually
python3.12 -m pip install langchain-core==0.3.79 --force-reinstall
python3.12 -m pip install langchain==0.3.0 --force-reinstall
python3.12 -m pip install langchain-community==0.3.0 --force-reinstall
python3.12 -m pip install "langchain-weaviate>=0.0.6,<0.1.0" --force-reinstall
python3.12 -m pip install "weaviate-client>=3.0.0,<4.0.0" --force-reinstall
python3.12 -m pip install "langchain-mistralai>=0.2.0,<1.0.0" --force-reinstall
python3.12 -m pip install langchain-huggingface==0.3.0 --force-reinstall
python3.12 -m pip install streamlit --force-reinstall
python3.12 -m pip install python-dotenv --force-reinstall  
python3.12 -m pip install mistralai --force-reinstall
python3.12 -m pip install pypdf --force-reinstall
python3.12 -m pip install sentence-transformers --force-reinstall
python3.12 -m pip install pdf2image --force-reinstall
python3.12 -m pip install python-docx --force-reinstall
python3.12 -m pip install fpdf --force-reinstall
python3.12 -m pip install opencv-python --force-reinstall
python3.12 -m pip install pytesseract --force-reinstall

echo "All packages have been attempted to install individually."
