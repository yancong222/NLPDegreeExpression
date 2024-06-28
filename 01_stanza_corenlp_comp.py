# -*- coding: utf-8 -*-
"""01-Stanza_CoreNLP_Comp.ipynb

@author: Yan Cong
Created on Nov, 2023

### Installing Stanza

Installing and importing Stanza are as simple as running the following commands:
"""

# Install stanza; note that the prefix "!" is not needed if you are running in a terminal
!pip install stanza

# Import stanza
import stanza

"""### Setting up Stanford CoreNLP
"""

# Download the Stanford CoreNLP package with Stanza's installation command
# This'll take several minutes, depending on the network speed
corenlp_dir = './corenlp'
stanza.install_corenlp(dir=corenlp_dir)

# Set the CORENLP_HOME environment variable to point to the installation location
import os
os.environ["CORENLP_HOME"] = corenlp_dir

# Examine the CoreNLP installation folder to make sure the installation is successful
!ls $CORENLP_HOME

import subprocess

import stanza
from stanza.protobuf import SemgrexRequest, SemgrexResponse
from stanza.server.client import resolve_classpath

stanza.download('en')
nlp = stanza.Pipeline('en')

nlp = stanza.Pipeline('en',processors='tokenize,pos,lemma,depparse')

"""## Custom functions"""

def send_request(request, response_type, java_main):
    """
    Use subprocess to run the Semgrex processor on the given request

    Returns the protobuf response
    """
    pipe = subprocess.run(["java", "-cp", resolve_classpath(), java_main],
                          input=request.SerializeToString(),
                          stdout=subprocess.PIPE)
    response = response_type()
    response.ParseFromString(pipe.stdout)
    return response

def send_semgrex_request(request):
    return send_request(request, SemgrexResponse,
                        "edu.stanford.nlp.semgraph.semgrex.ProcessSemgrexRequest")

def process_doc(doc, *semgrex_patterns):
    """
    Returns the result of processing the given semgrex expression on the stanza doc.

    Currently the return is a SemgrexResponse from CoreNLP.proto
    """
    request = SemgrexRequest()
    for semgrex in semgrex_patterns:
        request.semgrex.append(semgrex)

    for sent_idx, sentence in enumerate(doc.sentences):
        query = request.query.add()
        word_idx = 0
        for token in sentence.tokens:
            for word in token.words:
                query_token = query.token.add()
                query_token.word = word.text
                query_token.value = word.text
                if word.lemma is not None:
                    query_token.lemma = word.lemma
                if word.xpos is not None:
                    query_token.pos = word.xpos
                if word.upos is not None:
                    query_token.coarseTag = word.upos
                if token.ner is not None:
                    query_token.ner = token.ner

                node = query.graph.node.add()
                node.sentenceIndex = sent_idx+1
                node.index = word_idx+1

                if word.head != 0:
                    edge = query.graph.edge.add()
                    edge.source = word.head
                    edge.target = word_idx+1
                    edge.dep = word.deprel

                word_idx = word_idx + 1

    return send_semgrex_request(request)


def read_in_sentences(inputfile):
    sent_list = []
    line_list = []
    infile = pd.read_csv(data + inputfile)
    for i in infile.index:
      if i > 40338:
        row = infile['text'][i]
        #print('row: ', row)
        line_list.append([infile['answer_id'][i], row])
        #print('line_list: ', line_list)
        sent_list.append(row)
    return sent_list, line_list


def save_new_sent_list(found_flag,line_list,outputfile):
    with open(outputfile,'w') as outfile:
        for i in range(len(found_flag)):
            line_list_augmented = [found_flag[i]] + line_list[i]
            line_liststr = list(map(str,line_list_augmented))
            linestr = '\t'.join(line_liststr)
            outfile.write(linestr+'\n')
    return outputfile

"""# Starting here"""

from google.colab import drive
drive.mount('/content/drive')
data = ('/corpus_files/')

import pandas as pd
import numpy as np
import re
import math
import os
import string
import shutil, sys, glob

input = pd.read_csv(data+'file.csv', index_col = 0)
input_q1 = input[input.question_type_id.isin([1])] 
input_q1 = input_q1.dropna(subset = ['text'])

def sent_len(input):
  leng = len(input.split(' '))
  return leng

for i in input_q1.index:
  if type(input_q1['text'][i]) != float: # empty rows
    input_q1['text'][i] = input_q1['text'][i].split('.')

input_q1 = input_q1.explode(['text'])
input_q1['sentence_len'] = input_q1['text'].apply(lambda x: sent_len(x))
input_q1 = input_q1[((input_q1['sentence_len'] > 4) & (input_q1['sentence_len'] < 21))]

input_q1 = input_q1.reset_index(drop = True)
input_q1 = input_q1[['answer_id', 'anon_id', 'L1', 'gender', 'semester', 'placement_test',
       'course_id', 'level_id', 'class_id', 'question_id', 'version',
       'text_len', 'text', 'sentence_len']]
input_q1.tail()

input_q1.to_csv(data+'file.csv')

"""# Experiment"""

inputfile = 'file.csv' #187 problematic
semgrexrules = [

"{pos: JJR} > /obl:than/ {}", #"Alex is taller than Kai"
"{pos: RBR} > /obl:than/ {}", #"We marched faster than the truck"
"{pos: JJ} > advmod{pos: RBR} > /obl:than/ {}", #"We were more confused than her"
"{pos: RB} > advmod{pos: RBR} > /obl:than/ {}", #"We marched more quickly than the truck"

"{pos: JJR} > /ccomp/ {}", #"Alex is taller than she thought"
"{pos: JJR} > /ccomp|advcl:than/ {}", #"Alex did it better than she did"
"{pos: RBR} > /ccomp|advcl:than/ {}", #"Alex did it faster than she did"
"{pos: JJ} > advmod {pos:RBR} > /ccomp/ {}", #"Alex is more confused than she thought"
"{pos: RB} > advmod {pos:RBR} > /ccomp|advcl:than/ {}" #"Alex marched more quickly than she thought"

]

sent_list, line_list = read_in_sentences(inputfile)
found_flag = [0]*len(sent_list)
for i in range(len(sent_list)):
  if i%10 == 0:
    print(i)
  doc = nlp(sent_list[i])
  print('doc: ', doc)
  for j, rule in enumerate(semgrexrules):
    semgrexresults = process_doc(doc,rule)
    result_str = str(semgrexresults)
    print('here: ', result_str)
    print(sent_list[i])
    if result_str.count('match') > 0:
      found_flag[i] = str(j+1) # semgrex rules are numbered
      outputfile = inputfile.replace('.csv','.out')
      save_new_sent_list(found_flag,line_list,outputfile)
      print('saved: ', i)

inputfile = 'file.csv' #187 problematic

sent_list, line_list = read_in_sentences(inputfile)
found_flag = [0]*len(sent_list)
for i in range(len(sent_list)):
  if i%10 == 0:
    print(i)
  doc = nlp(sent_list[i])
  #print('doc: ', doc)
  for item in doc.iter_words():
    if item.feats == "Degree=Cmp":
      found_flag[i] = item.text
      outputfile = inputfile.replace('.csv','.out')
      save_new_sent_list(found_flag,line_list,outputfile)
  print('saved: ', i)
  shutil.move("/folder.out", "/folder")

for item in doc.iter_words():
  print(item)

doc.get('feats')

doc

"""# Runn the file"""

inputfile = 'file.csv'

sent_list, line_list = read_in_sentences(inputfile)
found_flag = [0]*len(sent_list)
for i in range(len(sent_list)):
  if i > 40338:
    if i%10 == 0:
      print(i)
    doc = nlp(sent_list[i])
    #print('doc: ', doc)
    for item in doc.iter_words():
      if item.feats == "Degree=Cmp":
        found_flag[i] = item.text
        outputfile = inputfile.replace('.csv','.out')
        save_new_sent_list(found_flag,line_list,outputfile)
    print('saved: ', i)
shutil.copy("/folder.out", "/files")




