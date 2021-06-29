# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os
import regex

# import tensorflow as tf
# import numpy as np
#
# from hyperparams import Hyperparams as hp
# from data_load import load_test_data, load_de_vocab, load_en_vocab
# from train import Graph
from nltk.translate.bleu_score import corpus_bleu

import pdb
import time


def get_result():
    with open("/home/lidong/tangle/muti_model/multi-data/test_2016.de", 'r', encoding="utf-8") as ref_in, open("results/results.txt", 'r', encoding="utf-8") as got_in:
        list_of_refs = []
        hypotheses = []
        for ref, got in zip(ref_in, got_in):

            # list_of_refs.append([_refine(ref)])
            # hypotheses.append(_refine(got))

            list_of_refs.append([ref.strip().split()])
            hypotheses.append(got.strip().split())
            # break
        score4 = corpus_bleu(list_of_refs, hypotheses)
        score3 = corpus_bleu(list_of_refs, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        score2 = corpus_bleu(list_of_refs, hypotheses, weights=(0.5, 0.5, 0, 0))
        score1 = corpus_bleu(list_of_refs, hypotheses, weights=(1, 0, 0, 0))
        print(score4, score3, score2, score1)


def _refine(line):
    line = regex.sub("<[^>]+>", "", line)
    line = regex.sub("[^\s\p{Latin}']", "", line)

    return [x for x in line.strip().split(" ") if x != ""]


if __name__ == '__main__':
    get_result()
