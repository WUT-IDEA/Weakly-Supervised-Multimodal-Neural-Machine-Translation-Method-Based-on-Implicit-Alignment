import tensorflow as tf
from graph_fusion import Graph
from hyperparams import Hyperparams as hp
from data_load import load_train_data, add_epoch
from tqdm import tqdm
import numpy as np

import os
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


def load_data(cut_size=0):
    X, Y, IMAGE_EN, IMAGE_DE = load_train_data()
    return X, Y, IMAGE_EN, IMAGE_DE

def get_random_half(X, Y, IMAGE_EN, IMAGE_DE):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    index_en = index[:int(len(index)/2)]
    index_de = index[int(len(index)/2):]
    x = [X[i] for i in index_en]
    y = [Y[i] for i in index_de]
    img_en = [IMAGE_EN[i] for i in index_en]
    img_de = [IMAGE_DE[i] for i in index_de]
    return x, y, img_en, img_de

        

def random_change(list):
    list_tmp = []
    list_return = []
    i = 0
    while list[i] != 0 and i < hp.maxlen:
        if np.random.uniform(0, 1) < (1-hp.Prob_Drop):
            list_tmp.append(list[i])
        i += 1

    index_orig = [i for i in range(len(list_tmp))]
    for i in range(len(index_orig)):
        index_orig[i] += np.random.uniform(0, hp.Word_Move_Dis)
    list_index = sorted(range(len(index_orig)), key=lambda k: index_orig[k])
    for i in list_index:
        list_return.append(list_tmp[i])
    for i in range(0, hp.maxlen - len(list_index)):
        list_return.append(0)
    return list_return


def train(enc_gate = False, dec_gate = False, img_dec_attention=False, use_coordinate=False):
    # Construct graph
    g = Graph(enc_gate = enc_gate, dec_gate = dec_gate, img_dec_attention=img_dec_attention, use_coordinate=use_coordinate)
    X_orig, Y_orig, IMAGE_EN_orig, IMAGE_DE_orig = load_data()
    batch = int(len(X_orig) / 1) // hp.batch_size_train
    print("Graph loaded")
    with g.graph.as_default():
        # Start session
        sv = tf.train.Supervisor(graph=g.graph,
                                 logdir=hp.logdir,
                                 save_model_secs=0,
                                 saver=tf.train.Saver(max_to_keep=0))
        with sv.managed_session() as sess:
            # writer = tf.summary.FileWriter("tf.log", sess.graph)
            for epoch in range(1, hp.num_epochs + 1):
                add_epoch()
                X, Y, IMAGE_EN, IMAGE_DE = get_random_half(X_orig, Y_orig, IMAGE_EN_orig, IMAGE_DE_orig)
                print("halfed")
                # if sv.should_stop(): break
                for i in tqdm(range(batch), total=batch, ncols=70, leave=False, unit='b'):
                    
                    # image_en = IMAGE_EN[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                    # image_de = IMAGE_DE[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                    # x = X[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                    # y = Y[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                    # x_random = [random_change(sent_vec) for sent_vec in x]
                    # y_random = [random_change(sent_vec) for sent_vec in y]
                    # x_orig = x
                    # y_orig = y



                    image_en = IMAGE_EN_orig[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                    image_de = IMAGE_DE_orig[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                    x_orig = X_orig[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                    y_orig = Y_orig[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                    x_random = x_orig
                    y_random = y_orig
                    x = x_orig
                    y = y_orig

                    
                    # train_op, mean_loss = sess.run([g.train_op, g.mean_loss], {g.x: x, g.y: y, g.image: image})
                    # print(mean_loss)
                    # train_op, loss_auto_en, loss_cycle_de, loss_cycle_en, loss_cycle_de, mean_loss, acc = sess.run([g.train_op, g.mean_loss_auto_en, g.mean_loss_cycle_de, g.mean_loss_cycle_en, g.mean_loss_cycle_de, g.mean_loss, g.acc], {g.x: x, g.y: y, g.x_random: x_random, g.y_random: y_random, g.image: image})
                    # print(loss_auto_en, loss_cycle_de, loss_cycle_en, loss_cycle_de, mean_loss, acc)
                    # train_op, loss1, loss2, loss3, loss4, acc = sess.run([g.train_op, g.mean_loss_auto_en, g.mean_loss_auto_de, g.mean_loss_cycle_en, g.mean_loss_cycle_de, g.acc], {g.x: x, g.y: y, g.x_random: x_random, g.y_random: y_random, g.image_en: image_en, g.image_de:image_de})
                    # print(loss1, loss2, loss3, loss4, acc) 

                    # train_op, loss_auto_en, loss_auto_de, loss_cycle_en, loss_cycle_de = sess.run([g.train_op, g.mean_loss_auto_en, g.mean_loss_auto_de, g.mean_loss_cycle_en, g.mean_loss_cycle_de], {g.x: x, g.y: y, g.x_random: x_random, g.y_random: y_random, g.image_en: image_en, g.image_de:image_de, g.x_orig:x_orig, g.y_orig:y_orig})
                    train_op, loss_supervised_en, loss_supervised_de, loss_auto_en, loss_auto_de, loss_cycle_en, loss_cycle_de = sess.run([g.train_op, g.mean_loss_supervised_en, g.mean_loss_supervised_de, g.mean_loss_auto_en, g.mean_loss_auto_de, g.mean_loss_cycle_en, g.mean_loss_cycle_de], {g.x: x, g.y: y, g.x_random: x_random, g.y_random: y_random, g.image_en: image_en, g.image_de:image_de, g.x_orig:x_orig, g.y_orig:y_orig})
                    # train_op, loss_auto_en, loss_auto_de, loss_cycle_en, loss_cycle_de = sess.run([g.train_op, g.mean_loss_auto_en, g.mean_loss_auto_de, g.mean_loss_cycle_en, g.mean_loss_cycle_de], {g.x: x, g.y: y, g.x_random: x_random, g.y_random: y_random, g.image_en: image_en, g.image_de:image_de, g.x_orig:x, g.y_orig:y})
                    # print(epoch)
                    print(loss_auto_en, loss_auto_de, loss_cycle_en, loss_cycle_de)
                    # merge_result = sess.run(g.merged)
                    # writer.add_summary(merge_result, 30*i)
                    
                    # acc = sess.run(g.acc)
                    # loss = sess.run(g.mean_loss)
                    # print('acc=%g, loss=%g' % (acc, loss))

                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/supervised2_epoch_%02d_gs_%d' % (epoch, gs))
                print('/supervised2_epoch_%02d_gs_%d' % (epoch, gs))
                
        #     writer.close()
        print("Done")






def train_fine_tune(enc_gate = False, dec_gate = False, img_dec_attention=False, use_coordinate=False):
    # Construct graph
    g = Graph(enc_gate = enc_gate, dec_gate = dec_gate, img_dec_attention=img_dec_attention, use_coordinate=use_coordinate)
    X, Y, IMAGE_EN, IMAGE_DE = load_data()
    batch = len(X) // hp.batch_size_train
    print("Graph loaded")
    with g.graph.as_default():
        # Start session
        sv = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sv.restore(sess, tf.train.latest_checkpoint(hp.pretrained_model))
        print("Restored!")
        for epoch in range(1, hp.num_epochs + 1):
            X, Y, IMAGE_EN, IMAGE_DE = get_random_half(X, Y, IMAGE_EN, IMAGE_DE)
            # if sv.should_stop(): break
            for i in tqdm(range(batch), total=batch, ncols=70, leave=False, unit='b'):
                x = X[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                image_en = IMAGE_EN[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                image_de = IMAGE_DE[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                y = Y[i * hp.batch_size_train: (i + 1) * hp.batch_size_train]
                # print(x)
                # print(y)
                x_random = [random_change(sent_vec) for sent_vec in x]
                y_random = [random_change(sent_vec) for sent_vec in y]
                # train_op, mean_loss = sess.run([g.train_op, g.mean_loss], {g.x: x, g.y: y, g.image: image})
                # print(mean_loss)
                # train_op, loss_auto_en, loss_cycle_de, loss_cycle_en, loss_cycle_de, mean_loss, acc = sess.run([g.train_op, g.mean_loss_auto_en, g.mean_loss_cycle_de, g.mean_loss_cycle_en, g.mean_loss_cycle_de, g.mean_loss, g.acc], {g.x: x, g.y: y, g.x_random: x_random, g.y_random: y_random, g.image: image})
                # print(loss_auto_en, loss_cycle_de, loss_cycle_en, loss_cycle_de, mean_loss, acc)
                train_op, loss1, loss2, loss3, loss4, acc = sess.run([g.train_op, g.mean_loss_auto_en, g.mean_loss_auto_de, g.mean_loss_cycle_en, g.mean_loss_cycle_de, g.acc], {g.x: x, g.y: y, g.x_random: x_random, g.y_random: y_random, g.image_en: image_en, g.image_de:image_de, g.epoch:epoch})
                print(loss1, loss2, loss3, loss4, acc)
                # merge_result = sess.run(g.merged)
                # writer.add_summary(merge_result, 30*i)
                
                # acc = sess.run(g.acc)
                # loss = sess.run(g.mean_loss)
                # print('acc=%g, loss=%g' % (acc, loss))
            gs = sess.run(g.global_step)
            sv.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
            print('/model_epoch_%02d_gs_%d' % (epoch, gs))
                
        #     writer.close()
        print("Done")


if __name__ == '__main__':
    if hp.use_pretrained_model:
        train_fine_tune(enc_gate = False, dec_gate = False, img_dec_attention=False, use_coordinate=False)
    else:
        train(enc_gate = False, dec_gate = False, img_dec_attention=False, use_coordinate=True)
