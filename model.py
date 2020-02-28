import time
import config as cfg
from preprocess import make_embedding_matrix, load_dict, decode_text, idx_to_text
from module import encoder_module, decoder_module
from batch_generator import batch_generator , load_training_data, load_val_data
import os
import tensorflow as tf
import random

from metrics.basic import Metrics
from metrics.bleu import BLEU
#from metrics.nll import NLL

# from metrics.acc import ACC


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
        self.idx2word_dict = load_dict()
        
        self.bleu = BLEU('BLEU', gram=[2,3,4,5])

            
        if init_train:
            self._init_train()
            train_week_stock, train_month_stock, t_month_stock,            train_input_cap_vector, train_output_cap_vector = load_training_data()
            self.total_iter = len(train_input_cap_vector)
            self.train_data = batch_generator(train_week_stock, train_month_stock, t_month_stock,                                              train_input_cap_vector, train_output_cap_vector, self.batch_size)
            self._init_eval()
            val_week_stock, val_month_stock, val_t_month_stock,             val_input_cap_vector, val_output_cap_vector = load_val_data()
            self.val_total_iter = len(val_input_cap_vector)
            self.val_data = batch_generator(val_week_stock, val_month_stock, val_t_month_stock,                                             val_input_cap_vector, val_output_cap_vector, self.batch_size)
            
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
                
            encoded_output, encoded_state = encoder_module(self.week_input,
                                                         self.month_input,
                                                         self.t_month_input,
                                                         self.encoder_layers,
                                                         self.decoder_layers,
                                                         self.hidden_dim)
            
            decoder_output, decoder_state = decoder_module(self.embedding_matrix,
                                                          encoded_output,
                                                          encoded_state,
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
            logits = decoder_output.rnn_output
            self.sample_id = decoder_output.sample_id
            self.loss =   tf.contrib.seq2seq.sequence_loss(logits=logits,
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
        
    # batch 단위로 계산
    def cal_metrics(self, infer_text, real_text):
        self.bleu.reset(infer_text = infer_text, real_text = real_text)
        return self.bleu.get_score()

    # bleu 등 계산을 위해. 
    def _init_eval(self):
        self.eval_graph = tf.Graph()
        with self.eval_graph.as_default():
            self.eval_week_input = tf.placeholder(tf.float64, shape= [None, 7])
            self.eval_month_input = tf.placeholder(tf.float64, shape=[None, 28])
            self.eval_t_month_input = tf.placeholder(tf.float64, shape=[None, 84])

            eval_encoded_output, eval_encoded_state = encoder_module(self.eval_week_input,
                                                                     self.eval_month_input,
                                                                     self.eval_t_month_input,
                                                                     self.encoder_layers,
                                                                     self.decoder_layers,
                                                                     self.hidden_dim)
            
            self.eval_decoder_output, eval_decoder_state = decoder_module(self.embedding_matrix,
                                                                      eval_encoded_output,
                                                                      eval_encoded_state,
                                                                      None, 
                                                                      self.decoder_layers, 
                                                                      self.hidden_dim,
                                                                      self.max_length, 
                                                                      self.vocab_size, 
                                                                      self.batch_size,
                                                                      self.dropout, 
                                                                      self.SOS_token, 
                                                                      self.EOS_token, 
                                                                      train = False)   
            self.eval_saver = tf.train.Saver()
        self.eval_session = tf.Session(graph=self.eval_graph,config=self.gpu_session_config())       
            
    def train_epoch(self, epochs):
        if not self.init_train:
            raise Exception('Train graph is not inited')
        with self.train_graph.as_default():
            if os.path.isfile(cfg.save_path + '.meta'):
                print('model restore..')
                self.train_saver.restore(self.train_session, cfg.save_path)
            else:
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
                    
                    _, loss, sample_id = self.train_session.run([self.optimizer, self.loss, self.sample_id], 
                                                            feed_dict = {self.week_input : week_stock,
                                                                         self.month_input : month_stock,
                                                                         self.t_month_input : t_month_stock,
                                                                         self.decoder_input : decoder_input,
                                                                         self.decoder_target : decoder_target})
                    total_loss += loss
                total_step += self.total_iter
                end = time.time()
                print('epoch: {}|{}  loss: {:.5f}   Time: {:.1f} min'.format(e+1, epochs, total_loss/total_step, (end-start_time)/60 ))
                if e % 50 ==0:
                    #self.train_saver.save(self.train_session, cfg.save_path)
                    sid = random.randint(0, self.batch_size-1)
                    #랜덤으로 하나 뽑고 
                    target_text = decode_text(decoder_target[sid],self.idx2word_dict)
                    output_text = decode_text(sample_id[sid],self.idx2word_dict)
                    print('sample text...')
                    print('target :' + target_text)
                    print('ifer :' + output_text)
                    self.eval()
    # bleu, NLL,acc
    def eval(self):
        with self.eval_graph.as_default():
            self.eval_saver.restore(self.eval_session, cfg.save_path)
            all_bleu = [0] * 4
            for step in range(self.val_total_iter//self.batch_size):
                data = next(self.val_data)
                week_stock = data['week_stock']
                month_stock = data['month_stock']
                t_month_stock = data['t_month_stock']

                pred_output = self.eval_session.run([self.eval_decoder_output], 
                                                        feed_dict = {self.eval_week_input : week_stock,
                                                                     self.eval_month_input : month_stock,
                                                                     self.eval_t_month_input : t_month_stock})   
                

                target_text = idx_to_text(data['decoder_input'][:,1:],self.idx2word_dict)
                output_text = idx_to_text(pred_output[0].sample_id,self.idx2word_dict)

                bleu_score = self.cal_metrics(target_text, output_text)
                for idx,score in enumerate(bleu_score):
                    all_bleu[idx] += score
                    
            for idx, bleu in enumerate(bleu_score):#2,3,4,5
                print('BLEU-{} : {}'.format(idx+2, bleu))
                    
                    
    def _init_infer(self):
            self.infer_graph = tf.Graph()
            with self.infer_graph.as_default():
                self.infer_week_input = tf.placeholder(tf.float64, shape = [None, 7])
                self.infer_month_input = tf.placeholder(tf.float64, shape = [None, 28])
                self.infer_t_month_input = tf.placeholder(tf.float64, shape = [None, 84])
                self.infer_encoded_output, self.infer_encoded_state = encoder_module(self.infer_week_input,
                                                                                    self.infer_month_input,
                                                                                    self.infer_t_month_input,
                                                                                    self.encoder_layers,
                                                                                    self.deocder_layers,
                                                                                    self.hidden_dim)
                self.infer_decoder_output, self.eval_decoder_state = decoder_module(self.embedding_matrix,
                                                                                   self.infer_encoded_output,
                                                                                   self.infer_encoded_state,
                                                                                   None,
                                                                                   self.decoder_layers,
                                                                                   self.hidden_dim,
                                                                                   self.max_length,
                                                                                   self.batch_size,
                                                                                   self.dropout,
                                                                                   self.SOS_token,
                                                                                   self.EOS_token,
                                                                                   train = False)
                self.infer_saver = tf.train.Saver()
            self.infer_session = tf.Session(graph = self.infer_graph, config = self.gpu_session_config())
            

