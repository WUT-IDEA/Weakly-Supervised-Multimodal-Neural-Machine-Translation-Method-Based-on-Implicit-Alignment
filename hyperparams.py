# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data

    source_train = '/data1/home/aha12315/Data_Unsupervised/val_val500.en'
    target_train = '/data1/home/aha12315/Data_Unsupervised/val_val500.de'
    train_img_en = '/data1/home/aha12315/Data_Unsupervised/val_features_512_val500.csv'
    train_img_de = '/data1/home/aha12315/Data_Unsupervised/val_features_512_val500.csv'
    
    # source_train = '/data1/home/aha12315/Data_Unsupervised/train_val500.en'
    # target_train = '/data1/home/aha12315/Data_Unsupervised/train_val500.de'
    # train_img_en = '/data1/home/aha12315/Data_Unsupervised/train_features_512_val500.csv'
    # train_img_de = '/data1/home/aha12315/Data_Unsupervised/train_features_512_val500.csv'

    source_test = '/data1/home/aha12315/Data_Unsupervised/test_2016_rep.en'
    target_test = '/data1/home/aha12315/Data_Unsupervised/test_2016_rep.de'
    
    test_img = '/data1/home/aha12315/Data_Unsupervised/test_features_512.csv'
    
    # training
    use_pretrained_model = False
    pretrained_model = 'tangle01'

    batch_size_train = 30 # alias = N
    batch_size_fusion = 1000
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'tangle03' # log directory 
    
    # model
    maxlen = 50 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 4 # number of encoder/decoder blocks
    num_epochs = 30
    num_heads = 8
    dropout_rate = 0.2
    Prob_Drop = 0.1
    Word_Move_Dis = 2
    sinusoid = False # If True, use sinusoid. If false, positional embedding.