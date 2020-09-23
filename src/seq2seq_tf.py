# encoding=utf-8
import tensorflow as tf
import random
import numpy as np


class Seq2seq:

    def __init__(self, logger, train_path, max_len, batch_size, epoch, loss, rate,
                 num_units, tf_config, model_path, summary_path, embedding_dim=300,
                 use_attention=True, use_teacher_forcing=True):
        self.logger = logger
        self.train_path = train_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.epoch = epoch
        self.loss = loss
        self.rate = rate
        self.num_units = num_units
        self.tf_config = tf_config
        self.model_path = model_path
        self.summary_path = summary_path
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        self.use_teacher_forcing = use_teacher_forcing

    def make_vocab(self, docs):
        w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
        i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
        for doc in docs:
            for w in doc:
                if w not in w2i:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)
        return w2i, i2w

    def get_input_feature(self):
        sources = []
        targets = []
        datas = []
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                s, t = line.replace('\n', '').split('\t')
                sources.append(list(s))
                targets.append(list(t))
        self.w2i_source, self.i2w_source = self.make_vocab(sources)
        self.w2i_target, self.i2w_target = self.make_vocab(targets)
        self.source_vocab_size = len(self.w2i_source)
        self.target_vocab_size = len(self.w2i_target)
        for i in range(len(sources)):
            source = sources[i]
            source_len = len(source)
            target = targets[i]
            target_len = len(target) + 1
            source_feature = [self.w2i_source[i] for i in source] + [self.w2i_source['_PAD']] * (240 - len(source))
            target_feature = [self.w2i_target[i] for i in target] + [self.w2i_target['_EOS']] + [self.w2i_target['_PAD']] * (250 - len(target))
            datas.append([source_feature, source_len, target_feature, target_len])
        return datas

    def batch_yield(self, datas, shuffle=False):
        '''
        产生batch数据
        :param data:
        :param shuffle:
        :return:
        '''
        if shuffle:
            random.shuffle(datas)
        sources, source_lens, targets, target_lens = [], [], [], []
        for data in datas:
            source = data[0]
            source_len = data[1]
            target = data[2]
            target_len = data[3]
            if len(sources) == self.batch_size:
                yield sources, source_lens, targets, target_lens
                sources, source_lens, targets, target_lens = [], [], [], []
            sources.append(source)
            source_lens.append(source_len)
            targets.append(target)
            target_lens.append(target_len)
        if len(sources) != 0:
            yield sources, source_lens, targets, target_lens

    def model(self, seq_inputs, seq_inputs_length, seq_targets, seq_targets_length):
        with tf.variable_scope('encoder'):
            encoder_embedding = tf.Variable(tf.random_uniform([self.source_vocab_size, self.embedding_dim]), dtype=tf.float32, name='encoder_embedding')
            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, seq_inputs)
            ((encoder_fw_outputs, encoder_bw_outputs),
             (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.GRUCell(self.num_units),
                cell_bw=tf.nn.rnn_cell.GRUCell(self.num_units),
                inputs=encoder_inputs_embedded,
                sequence_length=seq_inputs_length,
                dtype=tf.float32,
                time_major=False
            )
            encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)
            encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)
        with tf.variable_scope('decoder'):
            decoder_embedding = tf.Variable(
                tf.random_uniform([self.target_vocab_size, self.embedding_dim]),
                dtype=tf.float32,
                name='decoder_embedding'
            )
            tokens_go = tf.ones([self.batch_size], dtype=tf.int32, name='tokens_GO') * self.w2i_target["_GO"]
            if self.use_teacher_forcing:
                decoder_inputs = tf.concat([tf.reshape(tokens_go, [-1, 1]), seq_targets[:, :-1]], 1)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    tf.nn.embedding_lookup(decoder_embedding, decoder_inputs),
                    seq_targets_length
                )
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, tokens_go, self.w2i_target["_EOS"])
            decoder_cell = tf.nn.rnn_cell.GRUCell(self.num_units)
            if self.use_attention:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.num_units,
                    memory=encoder_outputs,
                    memory_sequence_length=seq_inputs_length
                )
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
                decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell,
                helper,
                decoder_initial_state,
                output_layer=tf.layers.Dense(self.target_vocab_size)
            )
            decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=tf.reduce_max(seq_targets_length)
            )
        return decoder_outputs.rnn_output
        # if self.use_beam_search > 1:
        #     self.out = decoder_outputs.predicted_ids[:, :, 0]
        # else:
        #     decoder_logits = decoder_outputs.rnn_output
        #     self.out = tf.argmax(decoder_logits, 2)
        #     sequence_mask = tf.sequence_mask(seq_targets_length, dtype=tf.float32)
        #     self.loss = tf.contrib.seq2seq.sequence_loss(
        #         logits=decoder_logits,
        #         targets=seq_targets,
        #         weights=sequence_mask
        #     )
        #     self.train_op = tf.train.AdamOptimizer(learning_rate=self.rate).minimize(self.loss)

    def fit(self):
        data = self.get_input_feature()
        seq_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='seq_inputs')
        seq_inputs_length = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32, name='seq_inputs_length')
        seq_targets = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='seq_targets')
        seq_targets_length = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32, name='seq_targets_length')
        logits = self.model(seq_inputs, seq_inputs_length, seq_targets, seq_targets_length)
        tf.add_to_collection('logits', logits)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=seq_targets,
            weights=tf.sequence_mask(seq_targets_length, dtype=tf.float32)
        )
        if 'sgd' == self.loss.lower():
            train_op = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        elif 'adam' == self.loss.lower():
            train_op = tf.train.AdamOptimizer(self.rate).minimize(loss)
        else:
            train_op = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for step, (sources, source_lens, targets, target_lens) in enumerate(self.batch_yield(data)):
                    _, curr_loss = sess.run([train_op, loss], feed_dict={seq_inputs: sources, seq_inputs_length: source_lens, seq_targets: targets, seq_targets_length: target_lens})
                    if step % 10 == 0:
                        self.logger.info('epoch:%d, batch: %d, current loss: %f' % (i, step+1, curr_loss))
            saver.save(sess, self.model_path)
            tf.summary.FileWriter(self.summary_path, sess.graph)

    def load(self, path):
        self.pred_sess = tf.Session(config=self.tf_config)
        saver = tf.train.import_meta_graph(path + '/model.meta')
        saver.restore(self.pred_sess, tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()
        self.seq_inputs = graph.get_tensor_by_name('seq_inputs:0')
        self.seq_inputs_length = graph.get_tensor_by_name('seq_inputs_length:0')
        self.seq_targets = graph.get_tensor_by_name('seq_targets:0')
        self.seq_targets_length = graph.get_tensor_by_name('seq_targets_length:0')
        self.decoder_outputs = tf.get_collection('decoder_outputs')

    def close(self):
        self.pred_sess.close()

    def _predict_text_process(self, text):
        seq = []
        for w in list(text):
            if w not in self.w2i_source:
                w = '_PAD'
            seq.append(self.w2i_source[w])
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
        else:
            seq += [self.w2i_source['_PAD']] * (self.max_len - len(seq))
        return seq

    def predict(self, text):
        seq = self._predict_text_process(text)
        pred, _ = self.pred_sess.run(self.decoder_outputs, feed_dict={self.seq_inputs: seq, self.seq_inputs_length: self.max_len})
        decoder_logits = pred.rnn_output
        out = tf.argmax(decoder_logits, 2)
        return out
