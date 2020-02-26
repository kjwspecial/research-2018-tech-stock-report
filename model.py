#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#커밋 새로 다시해!


# In[3]:


import time
import config as cfg
from preprocess import make_embedding_matrix
from module import encoder_module, decoder_module
from batch_generator import batch_generator , load_training_data, load_val_data

import tensorflow as tf


# In[ ]:


make_embedding_matrix


# In[20]:


class Model():
    def __init__(self, encoder_layers, decoder_layers, hidden_dim, batch_size, learning_rate, dropout, init_train = True):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.init_train = init_train
        #----fix----
        self.vocab_size = cfg.vocab_size
        self.max_length = cfg.max_length
        self.embedding_matrix = make_embedding_matrix(cfg.all_captions)
        self.SOS_token = cfg.SOS_token
        self.EOS_token = cfg.EOS_token
        
#         ##############for grad_cam####################
#         self.Y = tf.placeholder(tf.float32, [None, num_labels])
#         self.W = tf.Variable(initializer([hidden_dim, num_labels]))
#         self.b = tf.Variable(initializer([num_labels]))
#         encoded_output = tf.transpose(encoded_output, [1, 0, 2])
#         encoder_output = encoded_output[-1]
#         model = tf.matmul(encoder_output, W) + b
#         ##############################################
            
        if init_train:
            self._init_train()
            train_week_stock, train_month_stock, t_month_stock,            train_input_cap_vector, train_output_cap_vector = load_training_data()
            self.total_iter = len(train_input_cap_vector)
            self.train_data = batch_generator(train_week_stock, train_month_stock, t_month_stock,                                              train_input_cap_vector, train_output_cap_vector, self.batch_size)
            
       
    # gpu 탄력적으로 사용.
    def gpu_session_config(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return config
    
    def _init_train(self):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            with tf.variable_scope('encoder_input'):
                self.week_input = tf.placeholder(tf.float64, shape= [None, 7], name='week_input')
                self.month_input = tf.placeholder(tf.float64, shape=[None, 28], name='month_input')
                self.t_month_input = tf.placeholder(tf.float64, shape=[None, 84], name='t_month_input')

            with tf.variable_scope("decoder_input"):
                self.decoder_input = tf.placeholder(tf.int32, [None, self.max_length], name='decoder_input')
                self.decoder_target = tf.placeholder(tf.int32, [None, self.max_length], name='decoder_target')
                
            self.encoded_output, self.encoded_state = encoder_module(self.week_input,
                                                                     self.month_input,
                                                                     self.t_month_input,
                                                                     self.encoder_layers,
                                                                     self.decoder_layers,
                                                                     self.hidden_dim)
            
            self.decoder_outputs, self.decoder_state = decoder_module(self.embedding_matrix,
                                                                      self.encoded_output,
                                                                      self.encoded_state,
                                                                      self.decoder_input, 
                                                                      self.decoder_layers, 
                                                                      self.hidden_dim,
                                                                      self.max_length, 
                                                                      self.vocab_size, 
                                                                      self.batch_size,
                                                                      self.dropout, 
                                                                      self.SOS_token, 
                                                                      self.EOS_token, 
                                                                      train = True) 
            
            self.weights = tf.ones(shape=[self.batch_size, self.max_length],dtype=tf.float64)
            self.logits = self.decoder_outputs.rnn_output
            self.loss =   tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                      targets=self.decoder_target,
                                                      weights=self.weights,
                                                      average_across_timesteps=True,
                                                      average_across_batch=True)


            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients,5)
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params))
            self.train_init = tf.global_variables_initializer()
            self.train_saver = tf.train.Saver()
        self.train_session = tf.Session(graph=self.train_graph, config = self.gpu_session_config())
            
        #트레이닝 에펔
    def _init_eval():
        pass
    def eval_epoch():
        pass
    def _init_infer():
        pass
        
    def train_epoch(self, epochs):
        if not self.init_train:
            raise Exception('Train graph is not inited')
        with self.train_graph.as_default():
#             if path.isfile(self.model_file + '.meta'):
#                 self.train_saver.restore(self.train_session, self.model_file)
            #else:
            self.train_session.run(self.train_init)
            total_loss = 0
            total_step = 0
            start_time =time.time()
            for e in range(epochs):
                for step in range(self.total_iter// self.batch_size):
                    data = next(self.train_data)
                    week_stock = data['week_stock']
                    month_stock = data['month_stock']
                    t_month_stock = data['t_month_stock']
                    decoder_input = data['decoder_input']
                    decoder_target = data['decoder_target']
                    
                    _, loss,logits = self.train_session.run([self.optimizer, self.loss, self.logits], feed_dict = {self.week_input : week_stock,
                                                                                                                  self.month_input : month_stock,
                                                                                                                  self.t_month_input : t_month_stock,
                                                                                                                  self.decoder_input : decoder_input,
                                                                                                                  self.decoder_target : decoder_target})
                    total_loss += loss
                total_step += self.total_iter
                end = time.time()
                print('epoch: {}{}, \t loss : {}\t Time: {}'.format(e+1, epochs, total_loss/total_step, end-start_time))
                if e % 100 ==0:
                    self.train_saver.save(sess, './') #self.save_path)


# In[ ]:




