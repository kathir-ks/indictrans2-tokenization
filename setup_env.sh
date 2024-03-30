#!/bin/bash

# Install required packages
pip install nltk sacremoses pandas regex mock "transformers>=4.33.2" mosestokenizer
python3 -c "import nltk; nltk.download('punkt')"
pip install bitsandbytes scipy accelerate datasets
pip install sentencepiece sacrebleu

# Clone and install IndicTransTokenizer
git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer
pip install --editable ./
cd ..
