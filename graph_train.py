import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from modules import *


class Graph:
    def __init__(self, is_training=True, use_coordinate=False, enc_gate = False, dec_gate = False, img_dec_attention = False):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # if is_training:
            #     self.x, self.y, self.image, self.num_batch = get_batch_data()  # (N, T)
            #     self.image = tf.cast(self.image, tf.float32)
            # else:  # inference
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.x_random = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y_random = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.image_en = tf.placeholder(tf.float32, shape=(None, hp.maxlen, 512))
            self.image_de = tf.placeholder(tf.float32, shape=(None, hp.maxlen, 512))
            # self.image = tf.layers.dense(self.image, 512)

            # define decoder inputs
            self.decoder_inputs_en = tf.concat((tf.ones_like(self.x[:, :1]) * 2, self.x[:, :-1]), -1)  # 2:<S>
            self.decoder_inputs_de = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>

            # Load vocabulary
            de2idx, idx2de = load_de_vocab()
            en2idx, idx2en = load_en_vocab()


##############Auto-Encoding Loss#####################
            # Encoder-EN
            with tf.variable_scope("encoder-en", reuse=tf.AUTO_REUSE):
                ## Embedding
                self.enc_en = embedding(self.x_random,
                                     vocab_size=len(en2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed-en")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc_en += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe-en")
                else:
                    self.enc_en += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe-en")

                ## Dropout
                self.enc_en = tf.layers.dropout(self.enc_en,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-en_enc".format(i), reuse=tf.AUTO_REUSE):
                        ### Multihead Attention
                        self.enc_en = multihead_attention(queries=self.enc_en,
                                                       keys=self.enc_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="self_attention-en")

                        ### Feed Forward
                        self.enc_en = feedforward(self.enc_en, num_units=[4 * hp.hidden_units, hp.hidden_units],reuse=tf.AUTO_REUSE)



            # Decoder-EN
            with tf.variable_scope("decoder-en", reuse=tf.AUTO_REUSE):
                ## Embedding
                self.dec_en = embedding(self.decoder_inputs_en,
                                     vocab_size=len(en2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed-en")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec_en += positional_encoding(self.decoder_inputs_en,
                                                    vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe-en")
                else:
                    self.dec_en += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs_en)[1]), 0),
                                                  [tf.shape(self.decoder_inputs_en)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe-en")

                ## Dropout
                self.dec_en = tf.layers.dropout(self.dec_en,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-en_dec".format(i), reuse=tf.AUTO_REUSE):
                        ## Multihead Attention ( self-attention)
                        self.dec_en = multihead_attention(queries=self.dec_en,
                                                       keys=self.dec_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention-en")

                        ## Multihead Attention ( vanilla attention)
                        self.dec_en = multihead_attention(queries=self.dec_en,
                                                       keys=self.enc_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention-en")
                        if img_dec_attention:
                            self.dec_en = multihead_attention(queries=self.dec_en,
                                                           keys=self.image_en,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False,
                                                           scope="image_dec_attention-en")
                        ## Feed Forward
                        self.dec_en = feedforward(self.dec_en, num_units=[4 * hp.hidden_units, hp.hidden_units])



                        self.logits_dec_en = tf.layers.dense(self.dec_en, len(en2idx))
            self.preds_auto_en = tf.to_int32(tf.arg_max(self.logits_dec_en, dimension=-1), name="preds_auto_en")
            self.istarget_auto_en = tf.to_float(tf.not_equal(self.x, 0))
            self.acc_auto_en = tf.reduce_sum(tf.to_float(tf.equal(self.preds_auto_en, self.x)) * self.istarget_auto_en) / (
                tf.reduce_sum(self.istarget_auto_en))
            tf.summary.scalar('acc_auto_en', self.acc_auto_en)


            self.smoothed_auto_en = label_smoothing(tf.one_hot(self.x, depth=len(en2idx)))
            self.loss_auto_en = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_dec_en, labels=self.smoothed_auto_en)
            self.mean_loss_auto_en = tf.reduce_sum(self.loss_auto_en * self.istarget_auto_en) / (tf.reduce_sum(self.istarget_auto_en))
            tf.add_to_collection("losses", self.mean_loss_auto_en)







            # Encoder-DE
            with tf.variable_scope("encoder-de", reuse=tf.AUTO_REUSE):
                ## Embedding
                self.enc_de = embedding(self.y_random,
                                     vocab_size=len(de2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed-de")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc_de += positional_encoding(self.y,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe-de")
                else:
                    self.enc_de += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.y)[1]), 0), [tf.shape(self.y)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe-de")

                ## Dropout
                self.enc_de = tf.layers.dropout(self.enc_de,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-de_enc".format(i), reuse=tf.AUTO_REUSE):
                        ### Multihead Attention
                        self.enc_de = multihead_attention(queries=self.enc_de,
                                                       keys=self.enc_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="self_attention-de")

                        ### Feed Forward
                        self.enc_de = feedforward(self.enc_de, num_units=[4 * hp.hidden_units, hp.hidden_units],reuse=tf.AUTO_REUSE)



            # Decoder-DE
            with tf.variable_scope("decoder-de", reuse=tf.AUTO_REUSE):
                ## Embedding
                self.dec_de = embedding(self.decoder_inputs_de,
                                     vocab_size=len(de2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed-de")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec_de += positional_encoding(self.decoder_inputs_de,
                                                    vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe-de")
                else:
                    self.dec_de += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs_de)[1]), 0),
                                                  [tf.shape(self.decoder_inputs_de)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe-de")

                ## Dropout
                self.dec_de = tf.layers.dropout(self.dec_de,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-de_dec".format(i), reuse=tf.AUTO_REUSE):
                        ## Multihead Attention ( self-attention)
                        self.dec_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.dec_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention-de")

                        ## Multihead Attention ( vanilla attention)
                        self.dec_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.enc_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention-de")
                        if img_dec_attention:
                            self.dec_de = multihead_attention(queries=self.dec_de,
                                                           keys=self.image_de,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False,
                                                           scope="image_dec_attention-de")
                        ## Feed Forward
                        self.dec_de = feedforward(self.dec_de, num_units=[4 * hp.hidden_units, hp.hidden_units])

                        self.logits_dec_de = tf.layers.dense(self.dec_de, len(de2idx))
            self.preds_auto_de = tf.to_int32(tf.arg_max(self.logits_dec_de, dimension=-1), name="preds_auto_de")
            self.istarget_auto_de = tf.to_float(tf.not_equal(self.y, 0))
            self.acc_auto_de = tf.reduce_sum(tf.to_float(tf.equal(self.preds_auto_de, self.y)) * self.istarget_auto_de) / (
                tf.reduce_sum(self.istarget_auto_de))
            tf.summary.scalar('acc_auto_de', self.acc_auto_de)


            self.smoothed_auto_de = label_smoothing(tf.one_hot(self.y, depth=len(de2idx)))
            self.loss_auto_de = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_dec_de, labels=self.smoothed_auto_de)
            self.mean_loss_auto_de = tf.reduce_sum(self.loss_auto_de * self.istarget_auto_de) / (tf.reduce_sum(self.istarget_auto_de))
            tf.add_to_collection("losses", self.mean_loss_auto_de)





##############Cycle-Consistency Loss#####################

            # Encoder-EN
            with tf.variable_scope("encoder-en", reuse=True):
                ## Embedding
                self.enc_en = embedding(self.x_random,
                                     vocab_size=len(en2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed-en")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc_en += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe-en")
                else:
                    self.enc_en += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe-en")

                ## Dropout
                self.enc_en = tf.layers.dropout(self.enc_en,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-en_enc".format(i), reuse=True):
                        ### Multihead Attention
                        self.enc_en = multihead_attention(queries=self.enc_en,
                                                       keys=self.enc_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="self_attention-en")

                        ### Feed Forward
                        self.enc_en = feedforward(self.enc_en, num_units=[4 * hp.hidden_units, hp.hidden_units],reuse=True)

            

            # Decoder-DE
            with tf.variable_scope("decoder-de", reuse=True):
                ## Embedding
                self.dec_de = embedding(self.decoder_inputs_de,
                                     vocab_size=len(de2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed-de")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec_de += positional_encoding(self.decoder_inputs_de,
                                                    vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe-de")
                else:
                    self.dec_de += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs_de)[1]), 0),
                                                  [tf.shape(self.decoder_inputs_de)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe-de")

                ## Dropout
                self.dec_de = tf.layers.dropout(self.dec_de,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-de_dec".format(i), reuse=True):
                        ## Multihead Attention ( self-attention)
                        self.dec_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.dec_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention-de")

                        ## Multihead Attention ( vanilla attention)
                        self.dec_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.enc_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention-de")
                        if img_dec_attention:
                            self.dec_de = multihead_attention(queries=self.dec_de,
                                                           keys=self.image_de,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False,
                                                           scope="image_dec_attention-de")
                        ## Feed Forward
                        self.dec_de = feedforward(self.dec_de, num_units=[4 * hp.hidden_units, hp.hidden_units])




            # Encoder-DE
            with tf.variable_scope("encoder-de", reuse=True):
                ## Embedding
                self.enc_de = embedding(self.y_random,
                                     vocab_size=len(de2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed-de")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc_de += positional_encoding(self.y,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe-de")
                else:
                    self.enc_de += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.y)[1]), 0), [tf.shape(self.y)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe-de")

                ## Dropout
                self.enc_de = tf.layers.dropout(self.enc_de,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-de_enc".format(i), reuse=True):
                        ### Multihead Attention
                        self.enc_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.dec_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="self_attention-de")

                        ### Feed Forward
                        self.enc_de = feedforward(self.enc_de, num_units=[4 * hp.hidden_units, hp.hidden_units],reuse=True)

            # Decoder-EN
            with tf.variable_scope("decoder-en", reuse=True):
                ## Embedding
                self.dec_en = embedding(self.decoder_inputs_en,
                                     vocab_size=len(en2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed-en")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec_en += positional_encoding(self.decoder_inputs_en,
                                                    vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe-en")
                else:
                    self.dec_en += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs_en)[1]), 0),
                                                  [tf.shape(self.decoder_inputs_en)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe-en")

                ## Dropout
                self.dec_en = tf.layers.dropout(self.dec_en,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-en_dec".format(i), reuse=True):
                        ## Multihead Attention ( self-attention)
                        self.dec_en = multihead_attention(queries=self.dec_en,
                                                       keys=self.dec_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention-en")

                        ## Multihead Attention ( vanilla attention)
                        self.dec_en = multihead_attention(queries=self.dec_en,
                                                       keys=self.enc_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention-en")
                        if img_dec_attention:
                            self.dec_en = multihead_attention(queries=self.dec_en,
                                                           keys=self.image_en,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False,
                                                           scope="image_dec_attention-en")
                        ## Feed Forward
                        self.dec_en = feedforward(self.dec_en, num_units=[4 * hp.hidden_units, hp.hidden_units])

                        self.logits_dec_en = tf.layers.dense(self.dec_en, len(en2idx))
            self.preds_cycle_en = tf.to_int32(tf.arg_max(self.logits_dec_en, dimension=-1), name="preds_cycle_en")
            self.istarget_cycle_en = tf.to_float(tf.not_equal(self.x, 0))
            self.acc_cycle_en = tf.reduce_sum(tf.to_float(tf.equal(self.preds_cycle_en, self.x)) * self.istarget_cycle_en) / (
                tf.reduce_sum(self.istarget_cycle_en))
            tf.summary.scalar('acc_cycle_en', self.acc_cycle_en)


            self.smoothed_cycle_en = label_smoothing(tf.one_hot(self.x, depth=len(en2idx)))
            self.loss_cycle_en = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_dec_en, labels=self.smoothed_cycle_en)
            self.mean_loss_cycle_en = tf.reduce_sum(self.loss_cycle_en * self.istarget_cycle_en) / (tf.reduce_sum(self.istarget_cycle_en))
            tf.add_to_collection("losses", self.mean_loss_cycle_en)










            # Encoder-DE
            with tf.variable_scope("encoder-de", reuse=True):
                ## Embedding
                self.enc_de = embedding(self.y_random,
                                     vocab_size=len(de2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed-de")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc_de += positional_encoding(self.y,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe-de")
                else:
                    self.enc_de += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.y)[1]), 0), [tf.shape(self.y)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe-de")

                ## Dropout
                self.enc_de = tf.layers.dropout(self.enc_de,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-de_enc".format(i), reuse=True):
                        ### Multihead Attention
                        self.enc_de = multihead_attention(queries=self.enc_de,
                                                       keys=self.enc_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="self_attention-de")

                        ### Feed Forward
                        self.enc_de = feedforward(self.enc_de, num_units=[4 * hp.hidden_units, hp.hidden_units],reuse=True)

            # Decoder-EN
            with tf.variable_scope("decoder-en", reuse=True):
                ## Embedding
                self.dec_en = embedding(self.decoder_inputs_en,
                                     vocab_size=len(en2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed-en")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec_en += positional_encoding(self.decoder_inputs_en,
                                                    vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe-en")
                else:
                    self.dec_en += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs_en)[1]), 0),
                                                  [tf.shape(self.decoder_inputs_en)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe-en")

                ## Dropout
                self.dec_en = tf.layers.dropout(self.dec_en,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-en_dec".format(i), reuse=True):
                        ## Multihead Attention ( self-attention)
                        self.dec_en = multihead_attention(queries=self.dec_en,
                                                       keys=self.dec_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention-en")

                        ## Multihead Attention ( vanilla attention)
                        self.dec_en = multihead_attention(queries=self.dec_en,
                                                       keys=self.enc_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention-en")
                        if img_dec_attention:
                            self.dec_en = multihead_attention(queries=self.dec_en,
                                                           keys=self.image_en,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False,
                                                           scope="image_dec_attention-en")
                        ## Feed Forward
                        self.dec_en = feedforward(self.dec_en, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Encoder-EN
            with tf.variable_scope("encoder-en", reuse=True):
                ## Embedding
                self.enc_en = embedding(self.x_random,
                                     vocab_size=len(en2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed-en")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc_en += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe-en")
                else:
                    self.enc_en += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe-en")

                ## Dropout
                self.enc_en = tf.layers.dropout(self.enc_en,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-en_enc".format(i), reuse=True):
                        ### Multihead Attention
                        self.enc_en = multihead_attention(queries=self.dec_en,
                                                       keys=self.dec_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="self_attention-en")

                        ### Feed Forward
                        self.enc_en = feedforward(self.enc_en, num_units=[4 * hp.hidden_units, hp.hidden_units],reuse=True,)

            

            # Decoder-DE
            with tf.variable_scope("decoder-de", reuse=True):
                ## Embedding
                self.dec_de = embedding(self.decoder_inputs_de,
                                     vocab_size=len(de2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed-de")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec_de += positional_encoding(self.decoder_inputs_de,
                                                    vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe-de")
                else:
                    self.dec_de += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs_de)[1]), 0),
                                                  [tf.shape(self.decoder_inputs_de)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe-de")

                ## Dropout
                self.dec_de = tf.layers.dropout(self.dec_de,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-de_dec".format(i), reuse=True):
                        ## Multihead Attention ( self-attention)
                        self.dec_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.dec_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention-de")

                        ## Multihead Attention ( vanilla attention)
                        self.dec_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.enc_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention-de")
                        if img_dec_attention:
                            self.dec_de = multihead_attention(queries=self.dec_de,
                                                           keys=self.image_de,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False,
                                                           scope="image_dec_attention-de")
                        ## Feed Forward
                        self.dec_de = feedforward(self.dec_de, num_units=[4 * hp.hidden_units, hp.hidden_units])

                        self.logits_dec_de = tf.layers.dense(self.dec_de, len(de2idx))
            self.preds_cycle_de = tf.to_int32(tf.arg_max(self.logits_dec_de, dimension=-1), name="preds_cycle_de")
            self.istarget_cycle_de = tf.to_float(tf.not_equal(self.y, 0))
            self.acc_cycle_de = tf.reduce_sum(tf.to_float(tf.equal(self.preds_cycle_de, self.y)) * self.istarget_cycle_de) / (
                tf.reduce_sum(self.istarget_cycle_de))
            tf.summary.scalar('acc_cycle_de', self.acc_cycle_de)


            self.smoothed_cycle_de = label_smoothing(tf.one_hot(self.y, depth=len(de2idx)))
            self.loss_cycle_de = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_dec_de, labels=self.smoothed_cycle_de)
            self.mean_loss_cycle_de = tf.reduce_sum(self.loss_cycle_de * self.istarget_cycle_de) / (tf.reduce_sum(self.istarget_cycle_de))
            tf.add_to_collection("losses", self.mean_loss_cycle_de)      





            # Final linear projection
            # Encoder-EN
            with tf.variable_scope("encoder-en", reuse=True):
                ## Embedding
                self.enc_en = embedding(self.x,
                                     vocab_size=len(en2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed-en")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc_en += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe-en")
                else:
                    self.enc_en += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe-en")

                ## Dropout
                self.enc_en = tf.layers.dropout(self.enc_en,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-en_enc".format(i), reuse=True):
                        ### Multihead Attention
                        self.enc_en = multihead_attention(queries=self.enc_en,
                                                       keys=self.enc_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="self_attention-en")

                        ### Feed Forward
                        self.enc_en = feedforward(self.enc_en, num_units=[4 * hp.hidden_units, hp.hidden_units],reuse=True)

            
            # # Decoder-EN
            # with tf.variable_scope("decoder-en", reuse=tf.AUTO_REUSE):
            #     ## Embedding
            #     self.dec_en = embedding(self.decoder_inputs_en,
            #                          vocab_size=len(en2idx),
            #                          num_units=hp.hidden_units,
            #                          scale=True,
            #                          scope="dec_embed-en")

            #     ## Positional Encoding
            #     if hp.sinusoid:
            #         self.dec_en += positional_encoding(self.decoder_inputs_en,
            #                                         vocab_size=hp.maxlen,
            #                                         num_units=hp.hidden_units,
            #                                         zero_pad=False,
            #                                         scale=False,
            #                                         scope="dec_pe-en")
            #     else:
            #         self.dec_en += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs_en)[1]), 0),
            #                                       [tf.shape(self.decoder_inputs_en)[0], 1]),
            #                               vocab_size=hp.maxlen,
            #                               num_units=hp.hidden_units,
            #                               zero_pad=False,
            #                               scale=False,
            #                               scope="dec_pe-en")

            #     ## Dropout
            #     self.dec_en = tf.layers.dropout(self.dec_en,
            #                                  rate=hp.dropout_rate,
            #                                  training=tf.convert_to_tensor(is_training))

            #     ## Blocks
            #     for i in range(hp.num_blocks):
            #         with tf.variable_scope("num_blocks_{}-en_dec".format(i), reuse=tf.AUTO_REUSE):
            #             ## Multihead Attention ( self-attention)
            #             self.dec_en = multihead_attention(queries=self.dec_en,
            #                                            keys=self.dec_en,
            #                                            num_units=hp.hidden_units,
            #                                            num_heads=hp.num_heads,
            #                                            dropout_rate=hp.dropout_rate,
            #                                            is_training=is_training,
            #                                            causality=True,
            #                                            scope="self_attention-en")

            #             ## Multihead Attention ( vanilla attention)
            #             self.dec_en = multihead_attention(queries=self.dec_en,
            #                                            keys=self.enc_en,
            #                                            num_units=hp.hidden_units,
            #                                            num_heads=hp.num_heads,
            #                                            dropout_rate=hp.dropout_rate,
            #                                            is_training=is_training,
            #                                            causality=False,
            #                                            scope="vanilla_attention-en")
            #             if img_dec_attention:
            #                 self.dec_en = multihead_attention(queries=self.dec_en,
            #                                                keys=self.image,
            #                                                num_units=hp.hidden_units,
            #                                                num_heads=hp.num_heads,
            #                                                dropout_rate=hp.dropout_rate,
            #                                                is_training=is_training,
            #                                                causality=False,
            #                                                scope="image_dec_attention-en")
            #             ## Feed Forward
            #             self.dec_en = feedforward(self.dec_en, num_units=[4 * hp.hidden_units, hp.hidden_units])



            #             self.logits_dec_en = tf.layers.dense(self.dec_en, len(en2idx))
            
            # self.logits = self.logits_dec_en
            # self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1), name="preds")
            # self.istarget = tf.to_float(tf.not_equal(self.x, 0))
            # self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.x)) * self.istarget) / (
            #     tf.reduce_sum(self.istarget))
            # tf.summary.scalar('acc', self.acc)




            # Decoder-DE
            with tf.variable_scope("decoder-de", reuse=True):
                ## Embedding
                self.dec_de = embedding(self.decoder_inputs_de,
                                     vocab_size=len(de2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed-de")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec_de += positional_encoding(self.decoder_inputs_de,
                                                    vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe-de")
                else:
                    self.dec_de += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs_de)[1]), 0),
                                                  [tf.shape(self.decoder_inputs_de)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe-de")

                ## Dropout
                self.dec_de = tf.layers.dropout(self.dec_de,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}-de_dec".format(i), reuse=True):
                        ## Multihead Attention ( self-attention)
                        self.dec_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.dec_de,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention-de")

                        ## Multihead Attention ( vanilla attention)
                        self.dec_de = multihead_attention(queries=self.dec_de,
                                                       keys=self.enc_en,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention-de")
                        if img_dec_attention:
                            self.dec_de = multihead_attention(queries=self.dec_de,
                                                           keys=self.image_de,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False,
                                                           scope="image_dec_attention-de")
                        ## Feed Forward
                        self.dec_de = feedforward(self.dec_de, num_units=[4 * hp.hidden_units, hp.hidden_units])
                        self.logits_dec_de = tf.layers.dense(self.dec_de, len(de2idx))

            self.logits = self.logits_dec_de
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1), name="preds")
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)







            trainable_vars = tf.trainable_variables()
            freezed_var_list = [t for t in trainable_vars if not t.name.startswith(u'decoder-de_1') and not t.name.startswith(u'decoder-en_2')]
            # print(freezed_var_list)

            if is_training:
                # Loss
                # self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(de2idx)))
                # self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                # self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
                # self.variable_names = [v.name for v in tf.trainable_variables()]
                # print(self.variable_names)

                

                losses = tf.get_collection('losses')
                self.mean_loss = tf.add_n(losses)
                # self.mean_loss = self.mean_loss_auto_en + self.mean_loss_auto_de + self.mean_loss_cycle_en * self.mean_loss_cycle_en + self.mean_loss_cycle_de * self.mean_loss_cycle_de
                # self.mean_loss = tf.layers.dense(self.mean_loss_auto_en, 1) + tf.layers.dense(self.mean_loss_auto_de, 1) + tf.layers.dense(self.mean_loss_cycle_en, 1) + tf.layers.dense(self.mean_loss_cycle_de, 1)
                # self.mean_loss = self.mean_loss_auto_en

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step, var_list=freezed_var_list)

                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
                