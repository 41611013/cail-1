# -*- coding:utf8 -*-
import jieba
import pickle
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import time
from itertools import islice
import json
import os
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input,Flatten,Dense, concatenate
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adagrad,RMSprop, Adam,SGD
from keras.callbacks import EarlyStopping
import h5sparse
import gensim
import load_data as ld

train_df, valid_df, test_df, accus, laws = ld.load_data(data_type="big_data", data_pic=1, label_type="single")

jieba.load_userdict('user_dict.txt')  
# 创建停用词list  
def stopwordslist(filepath = './stop_word.txt'):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  


# 对句子进行分词  
def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip())  
    stopwords = stopwordslist('./stop_word.txt')  # 这里加载停用词的路径  
    outstr = ''  
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':  
                outstr += word  
                outstr += " "  
    return outstr

train_df['word_tok'] = [seg_sentence(text) for text in train_df['fact']]

with open('./temp_pkl_min/train.pkl', 'wb') as f:
    pickle.dump(train_df, f, -1)  
with open('./temp_pkl_min/train.pkl', 'rb') as train_f:
    train_df = pickle.load(train_f)

num_word = 20000
tokenizer = Tokenizer(num_words=num_word)
tokenizer.fit_on_texts(train_df['word_tok'])
word_index = tokenizer.word_index
word_index = {e:i for e,i in word_index.items() if i <= num_word}

with open('./temp_pkl_min/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./temp_pkl_min/tokenizer.pkl', 'rb') as e:
    tokenizer2 = pickle.load(e)
with open('./temp_pkl_min/word_index.pkl', 'wb') as f:
    pickle.dump(word_index,f, -1)
f.close()

facts = list(train_df['word_tok'])
sequence = tokenizer2.texts_to_sequences(facts) 
sequences = pad_sequences(sequence, maxlen=input_length, padding= 'post', truncating = 'post')

with open('./temp_pkl_min/sequences.pkl', 'wb') as f:
    pickle.dump(sequences,f, -1)
f.close()

with open('./temp_pkl_min/word_model.pkl', 'rb') as w2v:
    word_model = pickle.load(w2v)

embedding_matrix = np.zeros((len(word_index)+1, dim))
embedding_index = {}
count = 0
for word, i in word_index.items():
    try:
        embedding_vector = word_model[word]
        count = count +1
    except:
        embedding_vector = [0]*dim
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector 
        embedding_index[word] = embedding_vector
print(count)

with open('./temp_pkl_min/embedding_matrix.pkl', 'wb') as f:
    pickle.dump(embedding_matrix,f, -1)
f.close()

with open('./temp_pkl_min/train.pkl', 'rb') as train_f:
    train_df = pickle.load(train_f)
with open('./temp_pkl_min/embedding_matrix.pkl', 'rb') as embed_mtx:
    embedding_matrix = pickle.load(embed_mtx)
with open('./temp_pkl_min/word_model.pkl', 'rb') as w2v:
    word_model = pickle.load(w2v)
with open('./temp_pkl_min/word_index.pkl', 'rb') as word_index_f:
    word_index = pickle.load(word_index_f)
with open('./temp_pkl_min/tokenizer.pkl', 'rb') as e:
    tokenizer2 = pickle.load(e)
with open('./temp_pkl_min/sequences.pkl', 'rb') as sequences_f:
    sequences = pickle.load(sequences_f)

def load_arr_data(url):
    arr = []
    with open(url, encoding='utf-8') as arrfile:
        for line in arrfile:
            arr.append(str(line).strip())
    arr = process_arrdata(arr)
    return arr

def process_arrdata(arr):
    datas = {}
    for i,value in enumerate(arr):
        datas[value] = i
    return datas

accus = load_arr_data('./accu.txt')
laws = load_arr_data('./law.txt')

vec = []
label_appex = []

# data generate
import os 
import numpy as np
import json

list_dirs = os.walk('./sequences') 
for root, dirs, files in list_dirs: 
    for f in files: 
        path = os.path.join(root, f)
        with open(path) as fi: # Use file to refer to the file object
            for line in fi:
                vec.append(json.loads(line))
                label_appex.append(int(f.split('.')[0]))        
seq_appex = np.array(vec)

sequences2 = np.vstack((sequences,seq_appex))

# -*- coding:utf8 -*-

import jieba
import pickle
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import time
from itertools import islice
import json
import os
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adagrad,RMSprop, Adam,SGD
from keras.callbacks import EarlyStopping
import h5sparse
import gensim
from keras.utils.np_utils import to_categorical
from keras.utils.np_utils import to_categorical

#建立CNN模型

def cnn_build(label):
    embedding_layer = Embedding(len(word_index) + 1, embedding_matrix.shape[1],\
                                weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,\
                                trainable=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    filter_sizes = [3,4,5]
    convs = []
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=256, kernel_size=filter_size, padding='same', \
                        activation='relu')(embedded_sequences)
        l_pool = MaxPool1D(filter_size)(l_conv)
        convs.append(l_pool)
        
    x = Concatenate(axis=1)(convs)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(10)(x)
#     x = Dropout(0.2)(x)
#     x = Conv1D(64, 5, activation='relu')(x)
#     x = MaxPooling1D(5)(x)  # global max pooling
#     x = Dropout(0.6)(x)
#     x = Conv1D(128, 5, activation='relu')(x)
#     x = MaxPooling1D(5)(x)
#     x = Dropout(0.8)(x)
    x = Flatten()(x)    
    
#     x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x=BatchNormalization()(x)

    preds = Dense(label.shape[1], activation='softmax')(x)

    model = Model(sequence_input, preds)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    
    
    return model

def process_for_task(path, task):
    if task =='accu':
        print('accu')
        task_label = list(train_df['max_accu'])# + label_appex     
        label = to_categorical(task_label, len(set(accus)))
        model = cnn_build(label)
        MAX_SEQUENCE_LENGTH = sequences2.shape[1]
        model.fit(sequences, label, epochs=20, batch_size = 200, validation_split=0.2)
        accu_model_json = model.to_json()
        with open(path+"/accu_model.json", "w") as json_file:
            json_file.write(accu_model_json)
            # serialize weights to HDF5
            model.save_weights(path +"/accu_model.h5")
            print("Saved accu model to disk")
    if task =='law':
        print('law')
        task_label = train_df['max_law']
        label = to_categorical(task_label, len(set(laws)))
        MAX_SEQUENCE_LENGTH = sequences.shape[1]
        model = cnn_build(label)
        model.fit(sequences, label, epochs=20, batch_size = 200, validation_split=0.2)
        law_model_json = model.to_json()
        with open(path+"/law_model.json", "w") as json_file:
            json_file.write(law_model_json)
            # serialize weights to HDF5
            model.save_weights(path+"/law_model.h5")
            print("Saved law model to disk")
    if task =='time':
        print('time')
        task_label = train_df['time']
        label = to_categorical(task_label, len(set(train_df['time'])))
        MAX_SEQUENCE_LENGTH = sequences.shape[1]
        model = cnn_build(label)
        model.fit(sequences, label, epochs=20, batch_size = 200, validation_split=0.2)
        time_model_json = model.to_json()
        with open(path+"/time_model.json", "w") as json_file:
            json_file.write(time_model_json)
            # serialize weights to HDF5
            model.save_weights(path+"/time_model.h5")
            print("Saved time model to disk")
    return model  
    
    #accu
    # serialize model to JSON
    
ls = ['accu', 'law','time']
# ls = ['accu']
path = './model/cnn_model_3'
for i in ls:
    MAX_SEQUENCE_LENGTH = 500
    process_for_task(path, i)
