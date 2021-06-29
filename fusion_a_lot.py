import codecs
import os
import json
import time

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from graph_fusion import Graph
from nltk.translate.bleu_score import corpus_bleu

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


class Fusion:
    def __init__(self, model_list: list):
        self.model_list = model_list

    def eval(self, enc_gate = False, dec_gate = False, img_dec_attention = False, use_coordinate=False, use_img=True):
        g = Graph(is_training=False, enc_gate = enc_gate, dec_gate = dec_gate, img_dec_attention=img_dec_attention, use_coordinate=use_coordinate)
        print("Graph loaded")
        X, IMAGE_EN, IMAGE_DE, Sources, Targets = load_test_data(use_img=use_img)
        de2idx, idx2de = load_de_vocab()
        en2idx, idx2en = load_en_vocab()
        with g.graph.as_default():
            sv = tf.train.Saver()
            sess_list = [tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) for i in range(len(self.model_list))]
            for sess, model in zip(sess_list, self.model_list):
                sv.restore(sess, tf.train.latest_checkpoint(model))
            print("Restored!")
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/results.txt", "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                epoch = len(X) // hp.batch_size_fusion
                for i in range(epoch):
                    start = time.time()
                    ### Get mini-batches
                    x = X[i * hp.batch_size_fusion: (i + 1) * hp.batch_size_fusion]
                    image_en = IMAGE_EN[i * hp.batch_size_fusion: (i + 1) * hp.batch_size_fusion]
                    image_de = IMAGE_DE[i * hp.batch_size_fusion: (i + 1) * hp.batch_size_fusion]
                    sources = Sources[i * hp.batch_size_fusion: (i + 1) * hp.batch_size_fusion]
                    targets = Targets[i * hp.batch_size_fusion: (i + 1) * hp.batch_size_fusion]
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size_fusion, hp.maxlen), np.int32)
                    for j in tqdm(range(hp.maxlen)):
                        _logits_list = []
                        for sess in sess_list:
                            pred, logit = sess.run([g.preds, g.logits], {g.x: x, g.y: preds, g.image_en: image_en, g.image_de: image_de})
                            _logits_list.append(logit)
                        mean_logits = np.mean(_logits_list, 0)
                        _preds = np.argmax(mean_logits, axis=-1)
                        # print(j)
                        preds[:, j] = _preds[:, j]

                    for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                        got = " ".join(idx2de[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write(got + '\n')
                        ref = target.split()
                        hypothesis = got.split()
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)
                    batch_time = time.time() - start
                    print(
                        "i = {} / {}, time = {}s, remain = {}s".format(i, epoch, batch_time,
                                                                       (epoch - i) * batch_time))
                score = corpus_bleu(list_of_refs, hypotheses)
                with codecs.open('./' + model_list[0] + "/scores.txt", "a", "utf-8") as fout:
                    with open('./' + model_list[0] + "/checkpoint") as f:
                        lines = f.readlines()
                    fout.write(lines[0].strip().replace('model_checkpoint_path: ', '') + ': ' + str(score) + '\n')
        

                print("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    # model_list = ["tangle_test1","tangle_test2","tangle_test3","tangle_test4"]
    model_list = ["tangle03"]
    model = []
    with codecs.open('./' + model_list[0] + "/scores.txt", "w", "utf-8") as fout:
        pass
    def listdir(path, list_name):
        for file in os.listdir(path):
            if file.startswith('supervised2'):
                list_name.append(file.split('.')[0])
    listdir(model_list[0], model)
    model = list(set(model))
    model = sorted(model,key=lambda keys:[ord(i) for i in keys],reverse=False)
    if len(model) > 60:
        model = model[-60:]
    

    print(model)
    for name in model:
        with open('./' + model_list[0] + "/checkpoint") as f:
            lines = f.readlines()

        lines[0] = 'model_checkpoint_path: \"' + name + '\"\n'

        with open('./' + model_list[0] + "/checkpoint", "w") as f:
            f.writelines(lines)
        f = Fusion(model_list)
        f.eval(enc_gate = False, dec_gate = False, img_dec_attention=False, use_coordinate=True, use_img = True)



