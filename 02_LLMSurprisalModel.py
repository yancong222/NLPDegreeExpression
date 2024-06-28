# -*- coding: utf-8 -*-
"""
@author: Yan Cong
Created on Nov, 2023
"""

data = ('/files/')

import pandas as pd
import numpy as np
import re
import math
import os
import string
import shutil, sys, glob
import torch

!pip install transformers

!pip install sentencepiece

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, T5Model
T5_PATH = "t5-large" # "t5-small", "t5-base", "t5-large" [the best a CPU-computer can do, very slow], "t5-3b", "t5-11b"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

type(t5_tokenizer) 

def acceptability(seq):
  input_ids = t5_tokenizer.encode('cola sentence: ' + seq + ' </s>', return_tensors='pt')
  outputs = t5_mlm.generate(input_ids=input_ids)
  result = t5_tokenizer.decode(outputs[0])
  return result.strip('<pad>').strip('</s>').strip(' ')

df = pd.read_csv(data + 'file.csv', index_col = 0)
df['COLA_T5large'] = df['text'].apply(lambda x: acceptability(x))
df.head()

df.to_csv(data + 'file.csv')

df.columns

df_cleaned = pd.read_csv(data + 'file.csv', index_col = 0)
df_cmb = df_cleaned.merge(df, on = ['Cmp', 'answer_id', 'text', 'anon_id', 'L1', 'gender', 'semester',
       'placement_test', 'course_id', 'level_id', 'class_id', 'question_id',
       'version', 'text_len', 'sentence_len'], how = 'inner') 

df_cmb = df_cmb[df_cmb["text"].str.contains("Part of Speech") == False]
df_cmb = df_cmb[df_cmb["text"].str.contains("part of speech") == False]
df_cmb = df_cmb[df_cmb["text"].str.contains("Part of speech") == False]

df_cmb = df_cmb.reset_index(drop = True)
df_cmb = df_cmb.drop_duplicates(subset = ['text'], ignore_index = True)

df_cmb.tail()

df_cmb.to_csv(data + 'file.csv')

df = df_cmb
df['Cmp'].value_counts().to_csv(data+'count.csv')
df['L1'].value_counts().to_csv(data+'count.csv')
df['gender'].value_counts().to_csv(data+'count.csv')
df['level_id'].value_counts().to_csv(data+'count.csv')

df.COLA_T5large.value_counts()

"""# Target Surprisals GPTs"""

df = pd.read_csv(data + 'Cmp.csv', index_col = 0)
df.head(2)

!pip install minicons

from minicons import scorer
gpt2 = scorer.IncrementalLMScorer('gpt2', 'cpu')
distilgpt2 = scorer.IncrementalLMScorer('distilgpt2', 'cpu')
gptneo = scorer.IncrementalLMScorer('EleutherAI/gpt-neo-1.3B', 'cpu')

def target_surprisal(model, sentence, target):
  temp = []
  temp.append(sentence)
  surp = model.token_score(temp, surprisal = True, base_two = True)
  tuple_list = surp[0]
  # y is a tuple (word: surprisal)
  result = [y[1] for x, y in enumerate(tuple_list) if y[0] == target]
  # handle 'drives up' cases only if there is no exact match
  if len(result) == 0:
    result = [y[1] for x, y in enumerate(tuple_list) if y[0] in target]
  # take the average [also handle 'drives up' cases]
  return round(np.nanmean(result), 2)

df['gpt2_target_surprisal']=''
df['distilgpt2_target_surprisal']=''
df['gptneo_target_surprisal']=''
df.tail()

df['gpt2_target_surprisal'] = df.apply(lambda x: target_surprisal(gpt2, x.text, x.Cmp), axis=1)
df.to_csv(data + 'Cmp.csv')
print('finished: gpt2')

df['distilgpt2_target_surprisal'] = df.apply(lambda x: target_surprisal(distilgpt2, x.text, x.Cmp), axis=1)
df.to_csv(data + 'Cmp.csv')
print('finished: distilgpt2')

df['gptneo_target_surprisal'] = df.apply(lambda x: target_surprisal(gptneo, x.text, x.Cmp), axis=1)
df.to_csv(data + 'Cmp.csv')
print('finished: gptneo')

df.head()

"""# Sentence Surprisals GPTs"""

df = pd.read_csv(data + 'Cmp.csv', index_col = 0)
df['gpt2_len_tokens'] = ''
df['distilgpt2_len_tokens'] = ''
df['gptneo_len_tokens'] = ''
df['gpt2_tokens_surpscore'] = ''
df['distilgpt2_surpscore'] = ''
df['gptneo_surpscore'] = ''

df = df[df.L1.isin(['Arabic', 'Chinese', 'Spanish'])]
df = df.reset_index(drop = True)

df.tail(2)

def get_tokens_len_score(sent, model):
  input = []
  input.append(sent)
  output = model.token_score(input, surprisal = True, base_two = True)
  result = []
  result.append([len(output[0]), output[0]])
  return result

models = [('gptneo', gptneo), ('gpt2', gpt2), ('distilgpt2', distilgpt2)]
for model in models:
  df[model[0] + '_len_tokens'] = df['text'].apply(lambda x: get_tokens_len_score(x, model[1])[0][0])
  df[model[0] + '_tokens_surpscore'] = df['text'].apply(lambda x: get_tokens_len_score(x, model[1])[0][1])
  df.to_csv(data + 'file.csv')

df.head()

df.columns

df = df[['Cmp', 'answer_id', 'text', 'anon_id', 'L1', 'gender', 'semester',
       'placement_test', 'course_id', 'level_id', 'class_id', 'question_id',
       'version', 'text_len', 'sentence_len', 'COLA_T5large',
       'gpt2_target_surprisal', 'distilgpt2_target_surprisal',
       'gptneo_target_surprisal', 'gpt2_len_tokens', 'distilgpt2_len_tokens',
       'gptneo_len_tokens']]

def sentence_surprisal(stimuli, model):
  lst = []
  lst.append(stimuli)
  score = model.sequence_score(lst, reduction = lambda x: -x.mean(0).item())
  return round(score[0], 2)

df['gpt2_sent_surprisal'] = df['text'].apply(lambda x: sentence_surprisal(x, gpt2))
df.to_csv(data + 'file.csv')

df['distilgpt2_sent_surprisal'] = df['text'].apply(lambda x: sentence_surprisal(x, distilgpt2))
df.to_csv(data + 'file.csv')

df['gptneo_sent_surprisal'] = df['text'].apply(lambda x: sentence_surprisal(x, gptneo))
df.to_csv(data + 'file.csv')

df.head()

"""# Longitudinal"""

df['semester'].value_counts()

df_lg = df[['semester', 'anon_id']]
df_lg = df_lg.groupby(['semester'])
df_lg.head()

"""# Stats"""

from scipy.stats import pearsonr, spearmanr

df['gpt2_target_surprisal'].describe()

df['distilgpt2_target_surprisal'].describe()

df['gptneo_target_surprisal'].describe()

df[['placement_test', 'level_id', 'sentence_len', 'gpt2_target_surprisal', 'distilgpt2_target_surprisal', 'gptneo_target_surprisal']].corr()

spearmanr(df['level_id'], df['gpt2_target_surprisal']) 

spearmanr(df['level_id'], df['distilgpt2_target_surprisal'])

spearmanr(df['level_id'], df['gptneo_target_surprisal'])

df['COLA_T5large'].replace('acceptable', '1', inplace = True)
df['COLA_T5large'].replace('unacceptable', '0', inplace = True)
spearmanr(df['level_id'], df['COLA_T5large'])

df.to_csv(data + 'Cmp.csv')

