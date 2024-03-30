import json
from datasets import load_dataset
import argparse
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time

def split_into_sentences(text):

  # Define punctuation marks to split on
  punctuation = ".?!;:"
  sentences = []
  curr_sentence = ""
  l = len(text)
  for i, char in enumerate(text):

    curr_sentence += char

    if char in punctuation:
      if sentences and len(sentences[-1]) <=10:
        sentences[-1] += curr_sentence
        curr_sentence = ""
      else:
        sentences.extend([curr_sentence])
        curr_sentence = ""

  if curr_sentence:
    if sentences and len(sentences[-1]) <=10:
        sentences[-1] += curr_sentence
        curr_sentence = ""
    else:
        sentences.extend([curr_sentence])
        curr_sentence = ""

  return sentences

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def split_into_sentences(text):

  # Define punctuation marks to split on
  punctuation = ".?!;:"
  sentences = []
  curr_sentence = ""
  l = len(text)
  for i, char in enumerate(text):

    curr_sentence += char

    if char in punctuation:
      if sentences and len(sentences[-1]) <=10:
        sentences[-1] += curr_sentence
        curr_sentence = ""
      else:
        sentences.extend([curr_sentence])
        curr_sentence = ""

  if curr_sentence:
    if sentences and len(sentences[-1]) <=10:
        sentences[-1] += curr_sentence
        curr_sentence = ""
    else:
        sentences.extend([curr_sentence])
        curr_sentence = ""

  return sentences

def tokenize_sentences(sentences, tokenizer, ip, src_lang, tgt_lang):

    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)

    inputs = tokenizer(
          batch,
          src=True,
          truncation=True,
          padding="longest",
          return_tensors="pt",
          return_attention_mask=True,
    )

    inputs = {key: value.tolist() for key, value in inputs.items()}

    return inputs

parser = argparse.ArgumentParser(description="Tokenization script for splitting sentences and tokenizing.")
parser.add_argument("--subset", default="auto_math_text", type=str, help="Subset to process.")
parser.add_argument("--src_lang", default="eng_Latn", type=str, help="Source language.")
parser.add_argument("--tgt_lang", default="tam_Taml", type=str, help="Target language.")
parser.add_argument("--direction", default="en-indic", type=str, help="Translation direction.")
parser.add_argument("--shard", default=1, type=int, help="Subset Sharding Number.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size of tokenization.")

ṣubset = args.subset
src_lang = args.src_lang
tgt_lang = args.tgt_lang
direction = args.direction
shard = args.shard

# Parse arguments
args = parser.parse_args()

dataset = load_dataset("HuggingFaceTB/cosmopedia", subset)

ip = IndicProcessor(inference=True)
tokenizer = IndicTransTokenizer(direction=direction)

# Split and tokenize sentences, then write tokenized data to JSON files
MAX_SENTENCES_PER_FILE = 250000
file_counter = 1
batch_size = args.batch_size
sentences = []
indices = []
output_data = []

t = time.time()

for i, data_entry in enumerate(dataset['train']):
    index = i
    text = data_entry['text']
    sent = split_into_sentences(text)
    sentences.extend(sent)
    ind = [index] * len(sent)
    indices.extend(ind)

    if len(sentences) >= MAX_SENTENCES_PER_FILE:
        output_data = []
        for j in range(0, len(sentences), batch_size):
            batch_sentences = sentences[j : j + batch_size]
            batch_indices = indices[j : j + batch_size]
            tokenized_inputs = tokenize_sentences(batch_sentences, tokenizer, ip, src_lang, tgt_lang)
            output_data.append({"indices": batch_indices, "tokenized_input": tokenized_inputs})

        filename = f'{subset}_shard_{shard}_output_{file_counter}.json'
        write_json(output_data, filename)
        sentences = []
        indices = []
        file_counter += 1
        print(time.time() - t)
        t = time.time()

# Write remaining tokenized data to a JSON file
if sentences and indices:
        output_data = []
        for j in range(0, len(sentences), batch_size):
            batch_sentences = sentences[j : j + batch_size]
            batch_indices = indices[j : j + batch_size]

            tokenized_inputs = tokenize_sentences(batch_sentences, tokenizer, ip, src_lang, tgt_lang)
            output_data.append({"indices": batch_indices, "tokenized_input": tokenized_inputs})

        filename = f'{subset}_output_{file_counter}.json'
        write_json(output_data, filename)
        sentences = []
        indices = []
        file_counter += 1
