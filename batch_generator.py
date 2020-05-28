import config as cfg
from sklearn.utils import shuffle
import random

def batch_generator(week_stock, month_stock,t_month_stock,input_cap_vector, output_cap_vector, batch_size):
    total_size = len(input_cap_vector)
    sid = random.randint(0, total_size-1)
    input_cap, output_cap, week, month, t_month = shuffle(input_cap_vector,
                                                     output_cap_vector, 
                                                     week_stock,
                                                     month_stock,
                                                     t_month_stock,
                                                     random_state=sid)
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
