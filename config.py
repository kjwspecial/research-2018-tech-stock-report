import os

#-----Training param-----
encoder_layers = 3
decoder_layers = 3

hidden_dim = 128
batch_size = 256
dropout = 0.5
epochs = 7201
learning_rate = 2e-4
beam_width = 10

#----fixed-----
max_length = 32
vocab_size = 9099
SOS_token = 3
EOS_token = 4

#-----word embedding----
# $ git clone https://github.com/facebookresearch/fastText.git
# $ cd fastText
# $ mkdir build && cd build && cmake ..
# $ make && make install
#./fasttext skipgram -input ../../txt_datalysis_data.txt -output ../../fasttext_vector -dim 256 -ws 3 -epoch 50 -minCount 20


#-----preprocess------

stock_length = 25
per_num = 5
at_num = 5
minus_num = 5
plus_num = 5
enter_num = 5
u_dollor_num = 5
b_dollor_num = 0
#-----save path-----
save_path = os.getcwd()+'/save_model/stock_tech_model'

#-----data path-----
top100_stock_data_path = "./csv_data/top100_company_stock_close.csv"
stock_text_path = './csv_data/stock_text_concat_data/'
slang_path = './txt_data/slang.txt'
isdir = os.getcwd()+stock_text_path[1:]
is_train_dir = os.getcwd()+'/training_data/'
is_val_dir = os.getcwd()+'/val_data/'

idx2word_dict = './txt_data/idx2word_dict.txt'

fasttext = "../fasttext_vector.bin"

all_data = os.getcwd()+'/all_data/'
all_captions = os.getcwd()+'/captions/'
#-----training_data_path-----
week_stock = './training_data/train_week_stock.pickle'
month_stock = './training_data/train_month_stock.pickle'
t_month_stock = './training_data/train_t_month_stock.pickle'
train_input_cap_vector = './training_data/train_input_cap_vector.pickle'
train_output_cap_vector = './training_data/train_output_cap_vector.pickle'


#-----val_data_path-----
val_week_stock = './val_data/val_week_stock.pickle'
val_month_stock = './val_data/val_month_stock.pickle'
val_t_month_stock = './val_data/val_t_month_stock.pickle'
val_input_cap_vector = './val_data/val_input_cap_vector.pickle'
val_output_cap_vector = './val_data/val_output_cap_vector.pickle'