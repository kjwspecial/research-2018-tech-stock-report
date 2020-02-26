import os

#-----Training param-----
encoder_layers = 3
decoder_layers = 3

hidden_dim = 128
batch_size = 32
dropout = 0.2
epochs = 1201
learning_rate = 1e-4


#----fixed-----
max_length = 51
vocab_size = 13456
SOS_token = 2
EOS_token = 3

#-----word embedding----
# $ git clone https://github.com/facebookresearch/fastText.git
# $ cd fastText
# $ mkdir build && cd build && cmake ..
# $ make && make install
#./fasttext skipgram -input ../../txt_datalysis_data.txt -output ../../fasttext_vector -dim 256 -ws 3 -epoch 50 -minCount 20

#-----data path-----
top100_stock_data_path = "./csv_data/top100_company_stock_close.csv"
stock_text_path = './csv_data/stock_text_concat_data/'
slang_path = './txt_data/slang.txt'
isdir = os.getcwd()+stock_text_path[1:]
is_train_dir = os.getcwd()+'/training_data/'
is_val_dir = os.getcwd()+'/val_data/'

fasttext = "./fasttext_vector.bin"

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



