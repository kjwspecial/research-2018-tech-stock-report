import pandas as pd
import numpy as np
import pickle
import config as cfg
from sklearn.utils import shuffle


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
    with open(cfg.val_val_month_stock, 'rb') as f:
        val_month_stock = pickle.load(f)
    with open(cfg.val_t_month_stock, 'rb') as f:
        val_t_month_stock = pickle.load(f)
    with open(cfg.val_input_cap_vector, 'rb') as f:
        val_input_cap_vector = pickle.load(f)
    with open(cfg.val_output_cap_vector, 'rb') as f:
        val_output_cap_vector = pickle.load(f)
    return val_week_stock, val_month_stock, val_t_month_stock, val_input_cap_vector, val_output_cap_vector


def batch_generator(week_stock, month_stock,t_month_stock,input_cap_vector, output_cap_vector, batch_size):
    total_size = len(input_cap_vector)
    input_cap, output_cap, week, month, t_month = shuffle(input_cap_vector,
                                                     output_cap_vector, 
                                                     week_stock,
                                                     month_stock,
                                                     t_month_stock,
                                                     random_state=123)
    # Infinite loop.
    while True:
        batches = range(0, total_size - batch_size + 1, batch_size)
        for batch in batches:
            X_batch = input_cap[batch:batch + batch_size]
            Y_batch = output_cap[batch:batch + batch_size]
            week_batch = week[batch:batch + batch_size]
            month_batch = month[batch:batch + batch_size]
            t_month_batch = t_month[batch:batch + batch_size]
            x_data =             {
                'decoder_input': X_batch,
                'decoder_target': Y_batch,
                'week_stock': week_batch,
                'month_stock': month_batch,
                't_month_stock': t_month_batch
            }
            yield (x_data)

