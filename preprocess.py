import config as cfg
import pandas as pd
import numpy as np
from numpy import mean, std
import os

import tensorflow as tf
from gensim.models.wrappers.fasttext import FastText

from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import pickle
from functools import partial

top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True,
                                                  split=' ',
                                                  num_words=top_k, 
                                                  oov_token="<unk>",  
                                                  filters='!"#$%&()*+.,-:;=?@[\]^_`{|}~ ') 


def make_concat_stock(data_path, output_path):
    '''
        주가, 분석문 결합.
    '''
    df_stock = pd.read_csv(data_path)
    df_stock = df_stock.round(4)# 소수점 4째 자리까지 저장.
    company_name = df_stock.columns[1:].tolist()
    df_stock.sort_values(by='date', inplace=True, ascending=False)
    df_stock.reset_index(drop=True, inplace=True)
    stock_len=[7, 28, 84]
    for company in company_name:
        json_data = pd.read_json('json_data/$'+company+'.json')
        json_data.drop(['likes','replies','retweets','user_name','tweet_id','user_id','user_screen_name'],axis=1,inplace=True)#,'created_at'
        json_data.rename(columns={"created_at":"date"}, inplace = True)
        #date time ->str
        json_data['date']=json_data['date'].apply(lambda x : str(x))
        #시/분/초 날림
        json_data['date']=json_data['date'].apply(lambda x : str(x[:10]))
        
        json_data.sort_values(by='date', inplace=True, ascending=False)
        json_data.reset_index(drop=True, inplace=True)
        
        json_data[company+"_week"]=None
        json_data[company+"_month"]=None
        json_data[company+"_t_month"]=None
        for df_index in range(len(df_stock) - stock_len[2]):
            for index in range(len(json_data)):  
                if(json_data['date'][index]==df_stock['date'][df_index]):#날짜가 같으면                

                    week_stock = df_stock[company].iloc[df_index: df_index + stock_len[0]].tolist()
                    week_stock.reverse()
                    week_stock = ' '.join(str(stock) for stock in week_stock)
                    json_data[company+"_week"][index] = week_stock
                    # 4주
                    month_stock = df_stock[company].iloc[df_index: df_index + stock_len[1]].tolist()
                    month_stock.reverse()
                    month_stock = ' '.join(str(stock) for stock in month_stock)
                    json_data[company+"_month"][index] = month_stock            
                    # 8주
                    t_month_stock = df_stock[company].iloc[df_index: df_index + stock_len[2]].tolist()
                    t_month_stock.reverse()
                    t_month_stock = ' '.join(str(stock) for stock in t_month_stock)
                    json_data[company+"_t_month"][index] = t_month_stock
    
        json_data = json_data.dropna()
        json_data.to_csv(output_path+company+'.csv',index=None)
        print(company+'.csv saved..')

#잡단어 제거 목록
remove=['CEO','jamie','episode','email','member','development',' ai ','#patent','galaxies','Bezos','profits','arrayit','Invest',
        'BidaskScore','workspace','video','solarpower','Research','industries','MANAGEMENT','MERGER','#maxpain','#maxpain','options',
        'China','TICK-TOCK','watchlist','app ','platform','Software','join ','superstocks','Google+','GCP','Zhang','heatmap','article'
       ,'livestream','Rakuten','Frenkel','#crypto','ALTCOIN','BITCOIN','Morgan','Cramer','youtu.be','FinTwit','apps ','CapitanaMarvel',
       'discord','ceo','TipRanks⁩','finscreener','entertainment','shineestar','bitcoin']

def remove_text(text):
    for key in remove:
        if key in text:
            return None
    return text

# 분석문 관련 단어
keyword=['bull','bear','break','gap','resistance','still','breakout','run','back ','great','stagnation','uptrend','hold',
'helping' ,'held','huge','continuation','consolidation','upside','update','turn','high','breakdown','broke','nice','setting',
        'expansion','bounce','Pullback','buyback','move','swing','strong','carrying','shoulder','head','cup','downtrend','wedge','pennant']

def analysis_word(text):
    for key in keyword:
        if key in text:
            return text
    return None

#-------문장에 $없으면 제거--------------
def not_stock(text):
    if '$' in text:
        return text
    else:
        return None

#--------문장에 num< $개수 <num2 아니면 제거--------
def dollar_num(text,num,num2):
    count=0
    word = text.split(' ')
    for i in word:
        if '$' in i:
            count+=1
    if count>num and count<num2:
        return text
    else:
        return None
    
#--------문장에 @개수 num이상 제거--------
def at_num(text,num):
    count=0
    word = text.split(' ')
    for i in word:
        if '@' in i:
            count+=1
    if count<num:
        return text
    else:
        return None

#--------문장에 +개수 num이상 제거--------
def plus_num(text,num):
    count=0
    word = text.split(' ')
    for i in word:
        if '+' in i:
            count+=1
    if count<num:
        return text
    else:
        return None

    
#--------문장에 -개수 num이상 제거--------
def minus_num(text,num):
    count=0
    word = text.split(' ')
    for i in word:
        if '-' in i:
            count+=1
    if count<num:
        return text
    else:
        return None
    
#--------문장에 %개수 num이상 제거--------
def per_num(text,num):
    count=0
    word = text.split(' ')
    for i in word:
        if '%' in i:
            count+=1
    if count<num:
        return text
    else:
        return None
    
#--------문장에 엔터 num이상 많으면 제거--------
def enter_num(text,num):
    count=0
    word = text.split(' ')
    for i in word:
        if '\n' in i:
            count+=1
    if count<num:
        return text
    else:
        return None

#-------문장 길이 체크---------
def stock_length(text,num):
    word = text.split(' ')
    if len(word) >num or len(word)==1:
        return None
    return text


def preprocess_tweet_text(text):
    text = re.sub(r'http\S+',' ', str(text)) # Remove URLs
    text = re.sub("pic.twitter.*", ' ',text) # remove pic.twitter.*
    text = re.sub(r"http\S+",' ',text)       # html 제거
    
    text = text.lower() #소문자 변환

    text = re.sub('\'t',' not',text)
    text = re.sub('\'s',' is',text)
    
    text = re.sub('\$[A-Za-z0-9]+',' ',text)   # $태그 제거  ->혹은 그냥 영어 이외 제거
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text) # Remove @ mentions
    text = re.sub('#[A-Za-z]+',' ',text)       # #태그 제거
    
    text = re.sub("[^a-zA-Z]",' ',text) # 영어 이외 제거
    
    text = text.strip(" ")   # Remove whitespace
    text = re.sub(r' +', ' ', text) # Remove redundant spaces, 마지막에 해줘야함.

    # common HTML entities
    #text = re.sub(r'&lt;', '<', text)
    #text = re.sub(r'&gt;', '>', text)
    #text = re.sub(r'&amp;', '&', text)
    return text

def remove_stop_word(df):
    stop = stopwords.words("english") #up, down은 제외하자
    df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop) ]))
    return df

def stemming(df):
    tokenized_tweet = df.apply(lambda x: x.split())
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
    for i in tokenized_tweet.index:
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    return tokenized_tweet

def Lemmatizer(df):
    tokenized_tweet = df.apply(lambda x: x.split())
    Lemmat = WordNetLemmatizer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [Lemmat.lemmatize(i) for i in x]) # ,pos="v"
    for i in tokenized_tweet.index:
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    return tokenized_tweet


def short_len(df):
    df = df.apply(lambda x: ' '.join([word for word in x.split() if len(word)!=1 ]))
    return df

def drop_na(data):
    if 'nan' not in data:
        return data
    else:
        return None
    
def some_nan_bug(df):
    df['week_stock'] = df['week_stock'].apply(lambda x : (drop_na(x)))
    df['month_stock'] = df['month_stock'].apply(lambda x : (drop_na(x)))
    df['t_month_stock'] = df['t_month_stock'].apply(lambda x : (drop_na(x)))
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def slang_map(df):
    with open(cfg.slang_path) as file:
        slang_map = dict(map(str.strip, line.partition('\t')[::2]) 
        for line in file if line.strip())
    
    for i,sentence in enumerate(df['text']):
        sent=[]
        for word in sentence.split():
            if word in slang_map:
                word = slang_map[word]
            sent.append(word)
            text = " ".join(sent)
        df['text'].iloc[i]=text
    return df


def preprocess(df):
    df['text'] = df['text'].apply(lambda x: (stock_length(x,50))) #문장길이 제한 25
    df=df.dropna()
    df['text'] = df['text'].apply(lambda x: (not_stock(x))) #stock 아니면 제거
    df=df.dropna()  
    df['text'] = df['text'].apply(lambda x: (remove_text(x))) #잡단어 제거
    df=df.dropna()    
    df['text'] = df['text'].apply(lambda x: (per_num(x,5)))#% 5개 이상 제거   
    df=df.dropna()
    df['text'] = df['text'].apply(lambda x: (at_num(x,5)))#@ 5개 이상 제거  
    df=df.dropna()
    df['text'] = df['text'].apply(lambda x: (minus_num(x,5)))# - 5개 이상 제거   
    df=df.dropna()
    df['text'] = df['text'].apply(lambda x: (plus_num(x,5)))# + 5개 이상 제거
    df=df.dropna()
    df['text'] = df['text'].apply(lambda x: (enter_num(x,5)))# 엔터 5개 이상 제거
    df=df.dropna()
    df['text'] = df['text'].apply(lambda x: (analysis_word(x)))#특정단어 없으면 제거
    df=df.dropna()
    df['text'] = df['text'].apply(lambda x: dollar_num(x,0,20)) #$개수 1개
    df=df.dropna()
    df['text'] = df['text'].apply(lambda x: preprocess_tweet_text(x)) 
    df['text'] = df['text'].drop_duplicates()
    df=df.dropna()
    #df['text'] = remove_stop_word(df['text'])
    #short_len2(df['text'])
    df['text'] = short_len(df['text'])
    df['text'] = Lemmatizer(df['text'])
    df=df.dropna()
    #df= df.drop(columns=['date'])
    df.reset_index(drop=True, inplace=True)
    df.columns = ['date','text','week_stock','month_stock','t_month_stock']
    return df


def read_dir(data_path):
    all_file = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            all_file.append(file)
    return all_file


def all_company_preprocess(data_path, all_file):
    all_df = []
    for file_name in all_file:
        df = pd.read_csv(data_path+file_name)
        df = preprocess(df)
        all_df.append(df)
    result = pd.concat(all_df)
    result = result.dropna()
    return result


def std_nomal(stock_data):#각각의 정규화
    all_data=[]
    for i in stock_data:
        data_standadized_np= (i - mean(i, axis=0)) / std(i, axis=0)
        all_data.append(data_standadized_np)
    all_data =np.array(all_data)
    return all_data

def convert_float(stock_sentence): 
    stock_list = []
    for sentence in stock_sentence:
        source_tokens = sentence.split(' ')
        float_token = []
        for token in source_tokens:
            float_token.append(float(token))
        stock_list.append(float_token)
    return stock_list

def attach_symbol(text):
    train_captions =[]
    for sentence in text:
        train_captions.append('<s> '+sentence + ' </s>')
    return train_captions


def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def decoder_input(train_captions):
    tokenizer.fit_on_texts(train_captions)
    
    input_train_seq = []
    output_train_seq= []
    for caption in train_captions:
        input_train_seq.append(caption[:-5])
        output_train_seq.append(caption[4:])
    
    tokenizer.word_index['<pad>'] = 0
    input_train_seq=tokenizer.texts_to_sequences(input_train_seq)
    output_train_seq=tokenizer.texts_to_sequences(output_train_seq)
    max_length = calc_max_length(output_train_seq)
    
    input_cap_vector=tf.keras.preprocessing.sequence.pad_sequences(input_train_seq,maxlen= max_length,padding='post')
    output_cap_vector=tf.keras.preprocessing.sequence.pad_sequences(output_train_seq,maxlen= max_length,padding='post')
    
    return input_cap_vector, output_cap_vector

def make_embedding_matrix(train_captions):
    tokenizer.fit_on_texts(train_captions)
    model = FastText.load_fasttext_format(cfg.fasttext)
    #---------embedding matrix 만듬--------
    vocab_size = len(tokenizer.word_index)
    embedding_matrix = np.random.random((vocab_size, 256))
    for word,i in tokenizer.word_index.items(): # 1부터 시작함
        try:
            embedding_vector = model[word]
        except:
            #min count 이하 등장 단어
            #print(word, 'not found')
            pass
        if embedding_vector is not None:
            embedding_matrix[i-1] = embedding_vector
    return embedding_matrix


def make_data_set(df):
    stock_data = convert_float(df['t_month_stock'])
    stock_data = std_nomal(stock_data)
    captions = attach_symbol(df['text'])
    week_stock = stock_data[:,-7:]
    month_stock = stock_data[:,-28:]
    t_month_stock =stock_data
    
    input_cap_vector, output_cap_vector = decoder_input(captions)
    
    with open(cfg.all_data+'captions','wb') as f:
        pickle.dump(captions, f)
        
    train_week_stock, val_week_stock,    train_month_stock, val_month_stock,    train_t_month_stock, val_t_month_stock,    train_input_cap_vector, val_input_cap_vector,    train_output_cap_vector, val_output_cap_vector = train_test_split(week_stock, month_stock,                                                    t_month_stock, input_cap_vector, output_cap_vector,                                                   test_size=0.20, random_state=123)    
    if not os.path.isdir(cfg.is_train_dir):
        os.makedirs(cfg.is_train_dir)
        
    if not os.path.isdir(cfg.is_val_dir):
        os.makedirs(cfg.is_val_dir)
        
        
#-----training set-----
    with open(cfg.week_stock,'wb') as f:
        pickle.dump(train_week_stock, f)
        
    with open(cfg.month_stock,'wb') as f:
        pickle.dump(train_month_stock, f)
        
    with open(cfg.t_month_stock,'wb') as f:
        pickle.dump(train_t_month_stock, f)
        
    with open(cfg.train_input_cap_vector,'wb') as f:
        pickle.dump(train_input_cap_vector, f)

    with open(cfg.train_output_cap_vector,'wb') as f:
        pickle.dump(train_output_cap_vector, f)
        
#-----valdation set-----
    with open(cfg.val_week_stock,'wb') as f:
        pickle.dump(val_week_stock, f)
        
    with open(cfg.val_month_stock,'wb') as f:
        pickle.dump(val_month_stock, f)
        
    with open(cfg.val_t_month_stock,'wb') as f:
        pickle.dump(val_t_month_stock, f)
        
    with open(cfg.val_input_cap_vector,'wb') as f:
        pickle.dump(val_input_cap_vector, f)
        
    with open(cfg.val_output_cap_vector,'wb') as f:
        pickle.dump(val_output_cap_vector, f)        


def main():
    if not os.path.isdir(cfg.isdir):
        os.makedirs(cfg.isdir)
        print('#-----stock, text concat 작업...')
        make_concat_stock(cfg.top100_stock_data_path, cfg.stock_text_path)
    print("stock, text concat done...")
    print('#------Data preprocessing...')
    all_file = read_dir(cfg.stock_text_path)
    result = all_company_preprocess(cfg.stock_text_path, all_file)
    # nan bug
    result = some_nan_bug(result)
    #slang_map
    result = slang_map(result)
    result.to_csv('./all_data/all_data.csv',index = None)
    print('all_data.csv saved..')
    make_data_set(result)
    print('training data saved...')

if __name__ == '__main__':
    main()

