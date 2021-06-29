from __future__ import print_function
import codecs
import os
import json
import time

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

from os import listdir
from os.path import isfile, join


class Trans:
    def __init__(self, model_dir):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        ckpt = tf.train.get_checkpoint_state(model_dir)
        restore_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        restore_saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        self.graph = tf.get_default_graph()


def eval2(model1, model2):
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    g1 = Trans(model1)
    g2 = Trans(model2)
    if not os.path.exists('results'): os.mkdir('results')
    with codecs.open("results/results.txt", "w", "utf-8") as fout:
        list_of_refs, hypotheses = [], []
        epoch = len(X) // hp.batch_size
        for i in range(epoch):
            start = time.time()
            ### Get mini-batches
            x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
            sources = Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
            targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]
            ### Autoregressive inference
            preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
            for j in range(hp.maxlen):
                print(1)
                v1 = g1.graph.get_tensor_by_name("preds:0")
                # v2 = g1.graph.get_tensor_by_name("logits:0")
                t1 = g1.graph.get_tensor_by_name('x:0')
                t2 = g1.graph.get_tensor_by_name('y:0')
                print(1)
                _preds1 = g1.sess.run(v1, {t1: x, t2: preds})
                print(2)
                _preds2, _logits2 = g2.sess.run([g.preds, g.logits], {g.x: x, g.y: preds})
                mean_logits = np.mean([_logits1, _logits2], 0)
                _preds = np.argmax(mean_logits, axis=-1)
                print(j)
                preds[:, j] = _preds[:, j]

            for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                fout.write(got + "\n")

                print(source)
                print(target)
                print(got)

                # bleu score
                ref = target.split()
                hypothesis = got.split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)
            batch_time = time.time() - start
            print(
                "i = {} / {}, time = {}s, remain = {}s".format(i, epoch, batch_time,
                                                               (epoch - i) * batch_time))

        ## Calculate bleu score
        score = corpus_bleu(list_of_refs, hypotheses)
        print("Bleu Score = " + str(100 * score))
        g1.sess.close()
        g2.sess.close()


if __name__ == '__main__':
    eval2("tangle30", "tangle30")