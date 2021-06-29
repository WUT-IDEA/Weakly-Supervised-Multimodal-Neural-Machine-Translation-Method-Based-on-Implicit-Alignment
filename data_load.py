# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex
import pdb
import json


def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        # x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        # y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        x = [en2idx.get(word) for word in (source_sent + u" </S>").split() if
             word in en2idx]  # 1: OOV, </S>: End of Text
        y = [de2idx.get(word) for word in (target_sent + u" </S>").split() if word in de2idx]
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_img_data(in_path):
    first_line = False
    temp_list = []
    with codecs.open(in_path, 'r', 'utf-8') as f_in:
        for line in f_in:
            if first_line:
                first_line = False
                continue
            max_len_list = []
            vec = [float(x) for x in line.strip().split(",")]
            for i in range(hp.maxlen):
                max_len_list.append(vec)
            temp_list.append(max_len_list)
    data = np.array(temp_list, dtype=np.float)
    return data

def add_epoch():
    with open('epoch.txt', 'r') as f:
        lines = f.readlines()
        num = int(lines[0])
    with open('epoch.txt', 'w') as f:
        f.write(str(num + 1))
def load_epoch():
    print('lllllllllllllllllllload')
    with open('epoch.txt', 'r') as f:
        lines = f.readlines()
        return int(lines[0])

def load_train_data():
    en_sents = [_refine(line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if
                line and line[0] != "<"]
    de_sents = [_refine(line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if
                line and line[0] != "<"]
    images_en = load_img_data(hp.train_img_en)
    images_de = load_img_data(hp.train_img_de)
    # images_en = []
    # for i in range(28998):
    #     images_en.append([])
    # images_de = images_en
    X, Y, Sources, Targets = create_data(en_sents, de_sents)
    return X, Y, images_en, images_de


def load_test_data(use_img=True):
    en_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n")]
    de_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n")]
    if use_img:
        images_en = load_img_data(hp.test_img)
        images_de = load_img_data(hp.test_img)
    else:
        images_en = np.zeros((len(en_sents), 50, 512)).tolist()
        images_de = np.zeros((len(en_sents), 50, 512)).tolist()
    #     pdb.set_trace()
    X, Y, Sources, Targets = create_data(en_sents, de_sents)
    return X, images_en, images_de, Sources, Targets  # (1064, 150)


def _refine(line):
    line = regex.sub("<[^>]+>", "", line)
    line = regex.sub("[^\s\p{Latin}']", "", line)
    return line.strip()


def get_batch_data():
    # Load data
    X, Y, I = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    # X = tf.convert_to_tensor(X, tf.int32)
    # Y = tf.convert_to_tensor(Y, tf.int32)
    # I = tf.convert_to_tensor(I, tf.float32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y, I])

    # create batch queues
    x, y, i = tf.train.shuffle_batch(input_queues,
                                     num_threads=8,
                                     batch_size=hp.batch_size,
                                     capacity=hp.batch_size * 64,
                                     min_after_dequeue=hp.batch_size * 32,
                                     allow_smaller_final_batch=False)

    return x, y, i, num_batch  # (N, T), (N, T), ()

