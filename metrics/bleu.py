from multiprocessing import Pool 

import nltk
import os
import random
from nltk.translate.bleu_score import SmoothingFunction

from metrics.basic import Metrics


class BLEU(Metrics):
    def __init__(self, name=None, infer_text=None, real_text=None, gram=3):
        assert type(gram) == int or type(gram) == list, 'gram format error'
        super(BLEU, self).__init__('%s-%s'% (name, gram))
        self.infer_text = infer_text
        self.real_text = real_text
        self.gram = [gram] if type(gram) ==int else gram
        
    def get_score(self, is_fast = True, given_gram = None):
        if is_fast:
            return self.get_bleu_fast(given_gram)
        return self.get_bleu(given_gram)
    
    def get_bleu(self, given_gram=None): # given_gram : single gram일 경우 단어의 개수
        if given_gram is not None: # for single gram
            bleu = []
            reference = self.real_text
            weight = tuple((1. / given_gram for _ in range(given_gram))) # 단어 당 가중치 ex) (0.33 , 0.33, 0.33)
            for idx, hypothesis in enumerate(self.infer_text):
                bleu.append(self.cal_bleu(reference[idx],hypothesis, weight))
            return round(sum(bleu) / len(bleu),3)
        else: # multiple gram
            all_bleu =[]
            for ngram in self.gram:
                bleu =[]
                reference = self.real_text
                weight = tuple((1. / ngram for _ in range(ngram))) # gram 당 가중치.
                for idx, hypothesis in enumerate(self.infer_text):
                    bleu.append(self.cal_bleu(reference[idx], hypothesis, weight))
                all_bleu.append(round(sum(bleu) / len(bleu), 3))
            return all_bleu
        
    @staticmethod  
    def cal_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weight,smoothing_function=SmoothingFunction().method1)

    # dataset 초기화, batch단위로 할거임.
    def reset(self, infer_text = None, real_text = None):
        self.infer_text = infer_text if any(infer_text) else self.infer_text
        self.real_text = real_text if any(real_text) else self.real_text
    
    def get_bleu_parallel(self, ngram, reference):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = []
        for idx, hypothesis in enumerate(self.infer_text):
            result.append(pool.apply_async(self.cal_bleu, args = (reference[idx],hypothesis,weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt +=1
        pool.close()
        pool.join()
        return round(score / cnt, 3)
            
    def get_bleu_fast(self, given_gram = None):
        reference = self.real_text
        if given_gram is not None:
            return self.get_bleu_parallel(ngram=given_gram, reference = reference)
        else:
            all_bleu =[]
            for ngram in self.gram:
                all_bleu.append(self.get_bleu_parallel(ngram = ngram, reference =reference))
            return all_bleu

