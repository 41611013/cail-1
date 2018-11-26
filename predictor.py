import json
import numpy as np
# import thulac
from sklearn.externals import joblib
import keras
from keras.models import model_from_json
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 斯坦福分词，选取有意义的名词，专有名词，和动词
# from stanfordcorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP('/home/jeshe/stanford-corenlp-full-2017-06-09/', lang='zh')

# def seg_document(document):
#     result = [i for (i, j) in nlp.pos_tag(document) if j in ['NN', 'NR', 'VV']]
#     return result

import jieba

path = './cnn_model_1'
input_length = 500

class Predictor(object):
    def __init__(self):
        with open(path + '/tokenizer.pkl', 'rb') as e:
            self.tokenizer = pickle.load(e)

        # load json and create model
        json_file = open(path + '/accu_model.json', 'r')
        loaded_accu_model_json = json_file.read()
        json_file.close()
        self.accu_model = model_from_json(loaded_accu_model_json)
        # load weights into new model
        self.accu_model.load_weights(path + "/accu_model.h5")
        print("Loaded accu model from disk")

        json_file = open(path + '/law_model.json', 'r')
        loaded_law_model_json = json_file.read()
        json_file.close()
        self.law_model = model_from_json(loaded_law_model_json)
        # load weights into new model
        self.law_model.load_weights(path + "/law_model.h5")
        print("Loaded law model from disk")

        json_file = open(path + '/time_model.json', 'r')
        loaded_time_model_json = json_file.read()
        json_file.close()
        self.time_model = model_from_json(loaded_time_model_json)
        # load weights into new model
        self.time_model.load_weights(path + "/time_model.h5")
        print("Loaded time model from disk")

        #         self.law = joblib.load('predictor/model/law.model')
        #         self.accu = joblib.load('predictor/model/accu.model')
        #         self.time = joblib.load('predictor/model/time.model')
        self.batch_size = 1

    #         self.accumodel = joblib.load('predictor/model/accu.model')

    # self.cut = thulac.thulac(seg_only = True)

    def predict_law(self, vec):
        y = self.law_model.predict(vec)
        y = np.argmax(y, axis=1)
        #         return y
        return [y[0] + 1]

    def predict_accu(self, vec):
        y = self.accu_model.predict(vec)
        y = np.argmax(y, axis=1)
        #         return y
        return [y[0] + 1]

    def predict_time(self, vec):

        y = self.time_model.predict(vec)[0]
        y = np.argmax(y)

        # 返回每一个罪名区间的中位数
        if y == 0:
            return -2
        if y == 1:
            return -1
        if y == 2:
            return 120
        if y == 3:
            return 102
        if y == 4:
            return 72
        if y == 5:
            return 48
        if y == 6:
            return 30
        if y == 7:
            return 18
        else:
            return 6

    def predict(self, content):
        fact = []
        # print('text cut...')
        # print(content)
        for text in content:
            fact.append(' '.join(jieba.cut(text)))
        # print(fact)
        #         fact = self.cut.cut(content[0], text = True)
        #         print(fact[0])
        
        sequence = self.tokenizer.texts_to_sequences(fact) 
        vec = pad_sequences(sequence, maxlen=input_length, padding= 'post', truncating = 'post')
        # vec = self.tfidf.transform(fact)
        ans = {}

        ans['accusation'] = self.predict_accu(vec)
        ans['articles'] = self.predict_law(vec)
        ans['imprisonment'] = self.predict_time(vec)

        # print(ans)
        return [ans]
