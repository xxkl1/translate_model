#!/bin/zsh

# Setup script for language translation project

echo "====================================="
echo "Language Translation Project Setup"
echo "====================================="
echo ""

echo "Installing Python dependencies using Tsinghua mirror..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

echo ""
echo "Downloading Spacy language models..."
echo "This may take a few minutes..."

# Download commonly used language models
echo "Downloading German model..."
python3 -m spacy download de_core_news_sm

echo "Downloading English model..."
python3 -m spacy download en_core_web_sm

echo ""
echo "====================================="
echo "Setup complete!"
echo "====================================="
echo ""
echo "To use other languages, download the appropriate Spacy model:"
echo "  python3 -m spacy download <language_model>"
echo ""
echo "Available models:"
echo "  - French: fr_core_news_sm"
echo "  - Spanish: es_core_news_sm"
echo "  - Italian: it_core_news_sm"
echo "  - Portuguese: pt_core_news_sm"
echo "  - Dutch: nl_core_news_sm"
echo ""
echo "Now you can run the training:"
echo "  python3 main.py"
echo ""
echo "Or for a quick test:"
echo "  python3 main.py --dry_run --epochs 1"
