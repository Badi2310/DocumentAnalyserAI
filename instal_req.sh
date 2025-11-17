#!/bin/bash

# Bash script to install Python packages individually using pip with --force-reinstall
# This installs each package one by one, which can help with debugging or handling dependencies separately.
# If errors occur due to compatibility, you can add flags like --no-deps or --ignore-installed as needed.

# Update pip first (optional, but recommended)
pip install --upgrade pip

# Install each package individually
pip install langchain-core==0.3.79 --force-reinstall
pip install langchain==0.3.0 --force-reinstall
pip install langchain-community==0.3.0 --force-reinstall
pip install "langchain-weaviate>=0.0.6,<0.1.0" --force-reinstall
pip install "weaviate-client>=3.0.0,<4.0.0" --force-reinstall
pip install "langchain-mistralai>=0.2.0,<1.0.0" --force-reinstall
pip install langchain-huggingface==0.3.0 --force-reinstall
pip install streamlit --force-reinstall
pip install dotenv --force-reinstall
pip install mistralai --force-reinstall
pip install pypdf --force-reinstall
pip install sentence-transformers --force-reinstall

echo "All packages have been attempted to install individually."

