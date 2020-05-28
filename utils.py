import pickle
import config as cfg

def remove_sent_pad(target_text):
    for i,sent in enumerate(target_text):
        sentence = []
        for token in sent:
            if token == '<pad>':
                break
            sentence.append(token)
        target_text[i] = sentence
    return target_text


#문장 단위
def decode_text(sequence, vocab, end_token = '</s>'):
    result = []
    for idx in sequence:
        word = vocab[idx]
        if word == end_token:
            return ' '.join(result)
        result.append(vocab[idx])
    return ' '.join(result)


#batch단위
def idx_to_text(batch, vocab, end_token = '</s>'):
    results =[]
    for sentence in batch:
        sentence_token = []
        for idx in sentence:
            word = vocab[idx]
            if word == end_token:
                break
            sentence_token.append(word)
        if not sentence_token:# empty check, len==0
            sentence_token=['<pad>']
        results.append(sentence_token)
    return results


def load_dict():
    with open(cfg.idx2word_dict, 'r') as f:
        idx2word_dict = eval(f.read().strip())
    return idx2word_dict


def load_training_data():
    with open(cfg.week_stock, 'rb') as f:
        week_stock = pickle.load(f)
    with open(cfg.month_stock, 'rb') as f:
        month_stock = pickle.load(f)      
    with open(cfg.t_month_stock, 'rb') as f:
        t_month_stock = pickle.load(f)
    with open(cfg.train_input_cap_vector, 'rb') as f:
        train_input_cap_vector = pickle.load(f)
    with open(cfg.train_output_cap_vector, 'rb') as f:
        train_output_cap_vector = pickle.load(f)        
    return week_stock, month_stock, t_month_stock, train_input_cap_vector, train_output_cap_vector


def load_val_data():
    with open(cfg.val_week_stock, 'rb') as f:
        val_week_stock = pickle.load(f)
    with open(cfg.val_month_stock, 'rb') as f:
        val_month_stock = pickle.load(f)
    with open(cfg.val_t_month_stock, 'rb') as f:
        val_t_month_stock = pickle.load(f)
    with open(cfg.val_input_cap_vector, 'rb') as f:
        val_input_cap_vector = pickle.load(f)
    with open(cfg.val_output_cap_vector, 'rb') as f:
        val_output_cap_vector = pickle.load(f)
    return val_week_stock, val_month_stock, val_t_month_stock, val_input_cap_vector, val_output_cap_vector


def batch_seq_len(target_text):
    batch_seq_len =[]
    for text in target_text:
        cnt = 0
        for word in text:
            cnt += 1
            if word == 0:
                break
        batch_seq_len.append(cnt)
    return batch_seq_len



 