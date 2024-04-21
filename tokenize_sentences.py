'''
single process execution
'''


import json
from datasets import load_dataset
import argparse
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import nltk
nltk.download('punkt')


data_files = {
   "auto_math_text":{f"data/auto_math_text/train-{str(i).zfill(5)}-of-00018.parquet" for i in range(0, 18)},
   "combined":{"data/openstax/train-00000-of-00002.parquet",
               "data/openstax/train-00001-of-00002.parquet",
               "data/khanacademy/train-00000-of-00001.parquet", 
               "data/wikihow/train-00000-of-00002.parquet",
               "data/wikihow/train-00001-of-00002.parquet"
               },
  "stanford":{f"data/stanford/train-{str(i).zfill(5)}-of-00013.parquet" for i in range(0, 13)},
  "stories_shard_1":{f"data/stories/train-{str(i).zfill(5)}-of-00043.parquet" for i in range(0, 14)},
  "stories_shard_2":{f"data/stories/train-{str(i).zfill(5)}-of-00043.parquet" for i in range(14, 28)},
  "stories_shard_3":{f"data/stories/train-{str(i).zfill(5)}-of-00043.parquet" for i in range(28, 43)},
  "web_samples_v1_shard_1":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(0, 14)},
  "web_samples_v1_shard_2":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(14, 28)},
  "web_samples_v1_shard_3":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(28, 43)},
  "web_samples_v1_shard_4":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(43, 57)},
  "web_samples_v1_shard_5":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(57, 71)},
  "web_samples_v1_shard_6":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(71, 86)},
  "web_samples_v1_shard_7":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(86, 100)},
  "web_samples_v1_shard_8":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(100, 114)},
  "web_samples_v1_shard_9":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(114, 129)},
  "web_samples_v1_shard_10":{f"data/web_samples_v1/train-{str(i).zfill(5)}-of-00139.parquet" for i in range(129, 139)},
  "web_samples_v2_shard_1":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(0, 12)},
  "web_samples_v2_shard_2":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(12, 24)},
  "web_samples_v2_shard_3":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(24, 36)},
  "web_samples_v2_shard_4":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(36, 48)},
  "web_samples_v2_shard_5":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(48, 60)},
  "web_samples_v2_shard_6":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(60, 72)},
  "web_samples_v2_shard_7":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(72, 84)},
  "web_samples_v2_shard_8":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(84, 96)},
  "web_samples_v2_shard_9":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(96, 108)},
  "web_samples_v2_shard_10":{f"data/web_samples_v2/train-{str(i).zfill(5)}-of-00118.parquet" for i in range(108, 118)}
}


def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def split_into_sentences(index, text):

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

  return [[index] * len(sentences), sentences]


def tokenize_sentences(sentences,indices, tokenizer, ip, src_lang, tgt_lang):

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

    return {"indices": indices, "tokenized_input": inputs}


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Tokenize sentences')
    parser.add_argument("--subset", default=None, type=str, required=True, help=f"{data_files.keys()}")
    parser.add_argument("--src_lang", default="eng_Latn", type=str, required=False)
    parser.add_argument("--tgt_lang", default=None, type=str, required=True)
    parser.add_argument("--direction", default="en-indic", type=str, required=False)
    parser.add_argument("--tokenization_batch_size", default=64, type=int, required=True)
    parser.add_argument("--max_workers", default=96, type=int, required=True)


    args = parser.parse_args()

    subset = args.subset
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    direction = args.direction
    tokenization_batch_size = args.tokenization_batch_size
    max_workers = args.max_workers

    assert subset in data_files.keys()
    assert tgt_lang is not None

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction=direction)

    dataset = load_dataset("HuggingFaceTB/cosmopedia", data_files=data_files[subset])
    dataset = dataset['train']
    dataset = dataset['text']

    results = []

    if max_workers > 1:
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
             results.extend(executor.map(split_into_sentences, range(len(dataset)), dataset))

    else:
      for i, row in enumerate(dataset):
          result = split_into_sentences(i, row)
          results.append(result)
      
    indices = []
    sentences = []
    
    for result in results:   
        assert len(result[0])==len(result[1])
        indices.extend(result[0])
        sentences.extend(result[1])

    assert len(indices)==len(sentences)

    data = []

    if max_workers > 1:
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
              data.extend(executor.map(tokenize_sentences, (sentences[i : i + tokenization_batch_size] for i in range(0, len(sentences), tokenization_batch_size)),
                                      (indices[i : i + tokenization_batch_size] for i in range(0, len(indices), tokenization_batch_size)),
                                      repeat(tokenizer), repeat(ip), repeat(src_lang), repeat(tgt_lang)))
                
    else:
      for i in range(0, len(sentences), tokenization_batch_size):
        batch_sentences = sentences[i : i + tokenization_batch_size]
        batch_indices = indices[i : i + tokenization_batch_size]
        result = tokenize_sentences(batch_sentences, batch_indices, tokenizer, ip, src_lang, tgt_lang)
        data.append(result)

    file_name = f"{subset}.json"
    write_json(data, file_name)
    

