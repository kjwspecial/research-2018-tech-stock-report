import tensorflow as tf


def fc_layer(X, input_size):
    initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)
    w1 = tf.Variable(initializer([input_size, 1]), dtype=tf.float64)
    b1 = tf.Variable(tf.zeros(1, dtype=tf.float64))
    return tf.add(tf.matmul(X,w1),b1)


#같은 FC layer 연속으로 이어 붙임.
def make_encoder_input(inp, days, name):#"28days_fc_layer", "91days_fc_layer"
    cut = 7
    concat_fc_layer=[]
    for i in range(days//cut):#1주 단위로 끊음
        with tf.variable_scope(name,reuse=(i>=1)) as scope:
            concat_fc_layer.append(tf.nn.relu(fc_layer(inp[:, i*cut : (i+1)*cut], cut)))
    concat_fc_layer = tf.concat(concat_fc_layer, axis=1) #[None, 7]
    encoder_input = tf.expand_dims(concat_fc_layer,2)
    return encoder_input      


# make 3-module encoder state tuple 
def make_floor_state(week_state, month_state, three_month_state, layer_floor):
    encoded_state =[]
    encoded_state.append(tf.concat([week_state[(layer_floor)][0],
                       month_state[(layer_floor)][0],
                       three_month_state[(layer_floor)][0]],axis=1))

    encoded_state.append(tf.concat([week_state[(layer_floor)][1],
                       month_state[(layer_floor)][1],
                       three_month_state[(layer_floor)][1]],axis=1))
    encoded_state=tf.nn.rnn_cell.LSTMStateTuple(tf.zeros_like(encoded_state[0]),encoded_state[1])
    return encoded_state


def make_decoder_state(week_state, month_state, three_month_state, num_layers):
    state_list=[]
    for layer_num in range(num_layers):
        state_list.append(make_floor_state(week_state, month_state, three_month_state,layer_num))

    #( 'A' ,) + ('B', ) *(2) 튜플형태로 넣어줘야함
    if num_layers > 1:
        encoded_state = (tf.nn.rnn_cell.LSTMStateTuple(tf.zeros_like(state_list[0][0]), state_list[0][1]),)
        for i in range(1,num_layers):
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


def encoder_module(week_input, month_input, t_month_input, num_layers, de_num_layers ,hidden_dim):   

    # 3-type input
    w_encoder_input = make_encoder_input(week_input, 7, 'week_fc')
    m_encoder_input = make_encoder_input(month_input, 28, 'month_fc')
    tm_encoder_input = make_encoder_input(t_month_input, 84, 't_month_fc')
    
    # state.shape : [batch_size, step, hidden]
    w_encoder_out, w_encoder_state = encoder("week_encoder", w_encoder_input, num_layers, hidden_dim) 
    m_encoder_out, m_encoder_state = encoder("month_encoder", m_encoder_input, num_layers, hidden_dim)
    tm_encoder_out, tm_encoder_state =encoder("t_month_encoder", tm_encoder_input, num_layers, hidden_dim) 

    # 3-type state concat(LSTM)
    encoded_state = make_decoder_state(w_encoder_state, m_encoder_state, tm_encoder_state, de_num_layers)
                                       
    # ATTENTION 용도
    encoded_output=tf.concat([w_encoder_out, m_encoder_out, tm_encoder_out],axis=1) # batch x (time_step) x hidden_dim

    return encoded_output, encoded_state


def attention_decoder_cell(encoded_output, decoder_cell, vocab_size, hidden_dim, reuse=None):

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = hidden_dim*3, 
                                                                   memory=encoded_output,
                                                                   dtype=tf.float64)        
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell
                                                        ,attention_mechanism, 
                                                        attention_layer_size = hidden_dim*3 / 2,
                                                       alignment_history=True)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, vocab_size, reuse=reuse)
        
        return out_cell
     
        
def decode(encoded_output, encoded_state, num_layers, hidden_dim, vocab_size, batch_size, max_length, helper, keep_prob, scope_name, reuse=None):
    with tf.variable_scope(scope_name, reuse=reuse):
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
            
            
        out_cell = attention_decoder_cell(encoded_output, decoder_cell, vocab_size, hidden_dim, reuse= reuse)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell,
                                                  helper=helper
                                                  ,initial_state=out_cell.zero_state(dtype=tf.float64, batch_size=batch_size).clone(cell_state = encoded_state))
        
        
        outputs,final_state ,_= tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                    output_time_major=False,
                                                    impute_finished=True, 
                                                    maximum_iterations=max_length)
        return outputs, final_state
    

def decoder_module(embedding_matrix, encoded_output, encoded_state, decoder_input, num_layers, hidden_dim, max_length, vocab_size, batch_size, dropout, SOS_token, EOS_token, train = True):
    if decoder_input != None:
        keep_prob = 1 - dropout
    else:
        keep_prob = 1 
        
    inputs_embed = tf.nn.embedding_lookup(embedding_matrix, decoder_input) # batch_size  x max_length x embedding_dim  
    
    if train == True:
        helper = tf.contrib.seq2seq.TrainingHelper(inputs_embed,[max_length] * batch_size) 
        outputs, state = decode(encoded_output, encoded_state, num_layers, hidden_dim, vocab_size, batch_size, max_length, helper, dropout ,'decoder')
    else:    
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_matrix, 
                                                                    start_tokens=tf.tile([SOS_token], [batch_size]),
                                                                    end_token=EOS_token)    
        outputs, state = decode(encoded_output, encoded_state, num_layers, hidden_dim, vocab_size, batch_size,max_length, helper, 1 ,'decoder', reuse=True)

    return outputs, state


def _create_attention_images_summary(final_context_state):
    attention_images = (final_context_state.alignment_history.stack())
    # Reshape to (batch, src_seq_len, tgt_seq_len,1)
    attention_images = tf.expand_dims(
      tf.transpose(attention_images, [1, 2, 0]), -1)
    # Scale to range [0, 255]
    attention_images *= 255
    attention_summary = tf.summary.image("attention_images", attention_images)
    return attention_summary

