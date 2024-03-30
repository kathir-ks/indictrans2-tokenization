#!/bin/bash

# Install required packages
pip install nltk sacremoses pandas regex mock "transformers>=4.33.2" mosestokenizer
pip install bitsandbytes scipy accelerate datasets
pip install sentencepiece sacrebleu
pip install --upgrade huggingface_hub

huggingface-cli login --token "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"

# Clone and install IndicTransTokenizer
git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer
pip install --editable ./
cd ..
