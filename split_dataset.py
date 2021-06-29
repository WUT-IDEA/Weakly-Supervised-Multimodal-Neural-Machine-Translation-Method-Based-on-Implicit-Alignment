#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 18:51:15 2018

@author: congxin
"""

import codecs
import random

cn_file = "./corpora/cn.txt"
en_file = "./corpora/en.txt"

train_cn_file = "./corpora/train_cn"
train_en_file = "./corpora/train_en"
test_cn_file = "./corpora/test_cn"
test_en_file = "./corpora/test_en"

with codecs.open(cn_file, "r", "utf-8") as cn, codecs.open(en_file, "r", "utf-8") as en:
    cn_texts = cn.readlines()
    en_texts = en.readlines()

length = len(cn_texts)
ratio = 0.7
num = int(length * ratio)

train_idx = random.sample(range(length), num)
max_cn_len = 0
max_en_len = 0
with codecs.open(train_cn_file, "w", "utf-8") as train_cn, codecs.open(train_en_file, "w", "utf-8") as train_en, codecs.open(test_cn_file, "w", "utf-8") as test_cn, codecs.open(test_en_file, "w", "utf-8") as test_en:
    for i in range(length):
        if i in train_idx:
            train_cn.write(cn_texts[i])
            train_en.write(en_texts[i])
        else:
            test_cn.write(cn_texts[i])
            test_en.write(en_texts[i])
        
        if len(cn_texts[i]) > max_cn_len:
            max_cn_len = len(cn_texts[i])
            
        if len(en_texts[i]) > max_en_len:
            max_en_len = len(en_texts[i])
            
print(max_cn_len, max_en_len)
