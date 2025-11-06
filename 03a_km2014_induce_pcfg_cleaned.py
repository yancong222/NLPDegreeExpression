# -*- coding: utf-8 -*-
"""
Title: Second language learning of degree expressions: a computational approach
Author: Yan Cong, Purdue University
Date: July 2024
"""

# prep for file

import pandas as pd
import numpy as np
import os
import math
import csv
import shutil, sys

df = pd.read_csv(data + 'todo.csv', index_col = 0)
df.head()

import nltk
from nltk import CFG, Nonterminal, induce_pcfg
from nltk.parse.corenlp import CoreNLPParser

"""# Experiment with Stanza

![Latest Version](https://img.shields.io/pypi/v/stanza.svg?colorB=bc4545)
![Python Versions](https://img.shields.io/pypi/pyversions/stanza.svg?colorB=bc4545)

### Installing Stanza
"""

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

# experiment
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
doc = nlp('Barack Obama defeated him in Hawaii.')
print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')

stanza.download('en')
nlp = stanza.Pipeline('en')

import stanza
import nltk
from nltk import Nonterminal, induce_pcfg
from collections import defaultdict

# Function to recursively extract productions from Stanza's constituency parse
def extract_productions(tree, productions):
    if not isinstance(tree, nltk.Tree):
        return
    productions.extend(tree.productions())
    for subtree in tree:
        extract_productions(subtree, productions)

# Initialize the Stanza pipeline
nlp = stanza.Pipeline('en')

df['text'][0]

"""# Induce PCFG using Stanza and stanford corenlp parser"""

# Example sentences (replace with your 1000 sentences)
sentences = [
    #"the horse raced past the barn"
    #"here is another one"
    # Add more sentences here
    "our relationship was closer day by day"
]

# Parse sentences and extract productions
productions = []
for sentence in sentences:
    doc = nlp(sentence)
    for sent in doc.sentences:
        parse_tree = sent.constituency
        nltk_tree = nltk.Tree.fromstring(str(parse_tree))
        extract_productions(nltk_tree, productions)

# Ensure all productions have valid nonterminals
valid_productions = []
for prod in productions:
    lhs = prod.lhs()
    rhs = prod.rhs()
    if isinstance(lhs, nltk.Nonterminal) and all(isinstance(r, (nltk.Nonterminal, str)) for r in rhs):
        valid_productions.append(prod)

# Create a PCFG from the productions
pcfg = induce_pcfg(Nonterminal('ROOT'), valid_productions)

# Print the generated PCFG
print(pcfg)

# Optionally, save to a file
with open('pcfg_grammar.txt', 'w') as f:
    f.write(str(pcfg))

# Example sentences (replace with your 1000 sentences)
sentences = [
    #"the horse raced past the barn"
    #"here is another one"
    # Add more sentences here
    #"our house was eating"
    "one of my cousins whose girlfriend is my classmate is younger than me several months"
]

# Parse sentences and extract productions
productions = []
for sentence in sentences:
    doc = nlp(sentence)
    for sent in doc.sentences:
        parse_tree = sent.constituency
        nltk_tree = nltk.Tree.fromstring(str(parse_tree))
        extract_productions(nltk_tree, productions)

# Ensure all productions have valid nonterminals
valid_productions = []
for prod in productions:
    lhs = prod.lhs()
    rhs = prod.rhs()
    if isinstance(lhs, nltk.Nonterminal) and all(isinstance(r, (nltk.Nonterminal, str)) for r in rhs):
        valid_productions.append(prod)

# Create a PCFG from the productions
pcfg = induce_pcfg(Nonterminal('ROOT'), valid_productions)

# Print the generated PCFG
print(pcfg)