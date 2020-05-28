import tensorflow as tf
from tensorflow.python.layers.core import Dense
import config as cfg

def fc_layer(X, input_size):
    initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)
    w1 = tf.Variable(initializer([input_size, 1]), dtype=tf.float64)
    b1 = tf.Variable(tf.zeros(1, dtype=tf.float64))
    return tf.add(tf.matmul(X,w1),b1)

#같은 FC-layer 연속으로 이어 붙임
def make_encoder_input(inp, days, name):
    # 1 week
    cut = 7
    concat_fc_layer=[]
    #1주 단위로 끊음
    for i in range(days//cut):
        with tf.variable_scope(name,reuse=(i>=1)) as scope:
            concat_fc_layer.append(tf.nn.relu(fc_layer(inp[:, i*cut : (i+1)*cut], cut)))
    concat_fc_layer = tf.concat(concat_fc_layer, axis=1) #[None, 7]
    encoder_input = tf.expand_dims(concat_fc_layer,2)
    return encoder_input      


# 각 encoder module floor별로 state연결. 
def make_floor_state(week_state, month_state, three_month_state, layer_floor,hidden_dim):
    encoded_state =[]
    if len(week_state) == 1:
        layer_floor = 0
    encoded_state.append(tf.concat([week_state[layer_floor][0],
                       month_state[layer_floor][0],
                       three_month_state[layer_floor][0]],axis=1))

    encoded_state.append(tf.concat([week_state[layer_floor][1],
                       month_state[layer_floor][1],
                       three_month_state[layer_floor][1]],axis=1))
    
#Fc layer 이용해서 3 state => FC => 1 state dim으로 바꿔줌
#     initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)

#     with tf.variable_scope('reduce_cell_dim',reuse=tf.AUTO_REUSE) as scope:
#         w1 = tf.Variable(initializer([hidden_dim*3, hidden_dim]), dtype=tf.float64)
#         b1 = tf.Variable(tf.zeros(hidden_dim, dtype=tf.float64))
#         c = tf.add(tf.matmul(encoded_state[0],w1),b1)
            
#     with tf.variable_scope('reduce_hidden_dim',reuse=tf.AUTO_REUSE) as scope:       
#         w2 = tf.Variable(initializer([hidden_dim*3, hidden_dim]), dtype=tf.float64)
#         b2 = tf.Variable(tf.zeros(hidden_dim, dtype=tf.float64))
#         h = tf.add(tf.matmul(encoded_state[1],w2),b2)
        
#     encoded_state=tf.nn.rnn_cell.LSTMStateTuple(c,h) 

    encoded_state=tf.nn.rnn_cell.LSTMStateTuple(encoded_state[0],encoded_state[1])
    return encoded_state

# make decoder init state
def make_decoder_state(week_state, month_state, three_month_state, en_num_layers, de_num_layers,hidden_dim):
    state_list=[]
    for layer_num in range(de_num_layers):
        state_list.append(make_floor_state(week_state, month_state, three_month_state, layer_num,hidden_dim))

    #( 'A' ,) + ('B', ) *(2) 튜플형태로 넣어줘야함
    if de_num_layers > 1:
        encoded_state = (tf.nn.rnn_cell.LSTMStateTuple(tf.zeros_like(state_list[0][0]), state_list[0][1]),)
        for i in range(1,de_num_layers):
            if en_num_layers < i+1:
                encoded_state += (tf.nn.rnn_cell.LSTMStateTuple(tf.zeros_like(state_list[i][0]), tf.zeros_like(state_list[i][1])),)
            else:
                encoded_state += (tf.nn.rnn_cell.LSTMStateTuple(tf.zeros_like(state_list[i][0]), state_list[i][1]),)
    else:
        encoded_state=tf.nn.rnn_cell.LSTMStateTuple(tf.zeros_like(state_list[0][0]),state_list[0][1])
    return encoded_state


def multi_rnn_cell(num_layers,hidden_dim):
    cell_list = []
    for i in range(num_layers):
        cell_list.append(tf.nn.rnn_cell.LSTMCell(hidden_dim))
    cells = tf.contrib.rnn.MultiRNNCell(cell_list)
    return cells


def encoder(name, inputs, num_layers, hidden_dim):
    with tf.variable_scope(name):
        encoder_cell_list = multi_rnn_cell(num_layers, hidden_dim)
        encoder_list_output , encoder_list_state =tf.nn.dynamic_rnn(encoder_cell_list, inputs , dtype=tf.float64)
    return encoder_list_output, encoder_list_state


def encoder_module(week_input, month_input, t_month_input, en_num_layers, de_num_layers ,hidden_dim):   
    # 3-type input
    w_encoder_input = make_encoder_input(week_input, 7, 'week_fc')
    m_encoder_input = make_encoder_input(month_input, 28, 'month_fc')
    tm_encoder_input = make_encoder_input(t_month_input, 84, 't_month_fc')
    
    # state.shape : [batch_size, step, hidden]
    w_encoder_out, w_encoder_state = encoder("week_encoder", w_encoder_input, en_num_layers, hidden_dim) 
    m_encoder_out, m_encoder_state = encoder("month_encoder", m_encoder_input, en_num_layers, hidden_dim)
    tm_encoder_out, tm_encoder_state =encoder("t_month_encoder", tm_encoder_input, en_num_layers, hidden_dim) 
    
    # 3-type state concat(LSTM)
    encoded_state = make_decoder_state(w_encoder_state, m_encoder_state, tm_encoder_state, en_num_layers, de_num_layers,hidden_dim)
                                       
    # ATTENTION 용도 [batch_size, 1 + 4 + 12 ,128]
    encoded_output=tf.concat([w_encoder_out, m_encoder_out, tm_encoder_out],axis=1) # batch x (time_step) x hidden_dim   

    return encoded_output, encoded_state


def decoder_module(encoded_state, encoded_output, decoder_input, target_length, embedding_matrix, num_layers, hidden_dim, max_length, vocab_size, batch_size, dropout, SOS_token, EOS_token, train=True):
        if decoder_input != None:
            keep_prob = 1 - dropout
        else:
            keep_prob = 1 
        # make decoder 
        with tf.variable_scope("decoder_cell", reuse=tf.AUTO_REUSE):
            if num_layers > 1:
                cells = []
                for _ in range(num_layers):
                    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim*3)
                    decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell,
                                                                 input_keep_prob=keep_prob,
                                                                 output_keep_prob=keep_prob)
                    cells.append(decoder_cell)
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            else:
                decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim*3)
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob)            
        # output : vocab logit
        projection_layer = Dense(vocab_size, use_bias=False)
        
        if train == True:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = hidden_dim*3, 
                                                                   memory = encoded_output,
                                                                   dtype=tf.float64)        
            
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell
                                                        ,attention_mechanism, 
                                                        attention_layer_size = hidden_dim*3,
                                                       alignment_history=True)
            inputs_embed = tf.nn.embedding_lookup(embedding_matrix, decoder_input) # batch_size  x max_length x embedding_dim)
            helper = tf.contrib.seq2seq.TrainingHelper(inputs_embed,[max_length] * batch_size) 
            
            initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float64).clone(cell_state = encoded_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,
                                                      helper = helper,
                                                      initial_state = initial_state,
                                                      output_layer = projection_layer)
        #infer beam_search
        else:
            tiled_encoder_output = tf.contrib.seq2seq.tile_batch(encoded_output, multiplier=cfg.beam_width)
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoded_state, multiplier=cfg.beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = hidden_dim*3, 
                                                                    memory = tiled_encoder_output,
                                                                    dtype=tf.float64) 
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism, 
                                                               attention_layer_size = hidden_dim*3,
                                                               alignment_history=True)
            initial_state = decoder_cell.zero_state(dtype=tf.float64, batch_size=batch_size * cfg.beam_width)
            initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = decoder_cell,
                                                            embedding = embedding_matrix,
                                                            start_tokens = tf.tile([SOS_token], [batch_size]),
                                                            end_token = EOS_token,
                                                            initial_state = initial_state,
                                                            beam_width = cfg.beam_width,
                                                            output_layer = projection_layer,
                                                            length_penalty_weight=0.0,
                                                            coverage_penalty_weight=0.0) 

        outputs, final_state, _= tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                    output_time_major=False,
                                                    impute_finished=False,
                                                    maximum_iterations=tf.round(tf.reduce_max(target_length)) * 2)

        return outputs, final_state



def _create_attention_images_summary(final_context_state):
    attention_images = (final_context_state.alignment_history.stack())
    # Reshape to (batch, src_seq_len, tgt_seq_len,1)
    attention_images = tf.expand_dims(
      tf.transpose(attention_images, [1, 2, 0]), -1)
    # Scale to range [0, 255]
    attention_images *= 255
    attention_summary = tf.summary.image("attention_images", attention_images)
    return attention_summary