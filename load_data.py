import json
import pandas as pd
import numpy as np
import jieba
import hashlib


def load_data(data_type="small_data", data_pic=0.1, label_type="more"):
    #data_type: small_data / big_data
    #data_pic: 0-1 取数比例
    #label_type: more / single
    accus = load_arr_data('/CAIL/'+data_type+'/accu.txt')
    laws = load_arr_data('/CAIL/'+data_type+'/law.txt')
    train_df, all_accu_map, all_law_map = load_json_data('/CAIL/'+data_type+'/data_train.json', accus, laws, label_type)
    valid_df, all_accu_map, all_law_map = load_json_data('/CAIL/'+data_type+'/data_valid.json', accus, laws, label_type, all_accu_map, all_law_map)
    test_df, all_accu_map, all_law_map = load_json_data('/CAIL/'+data_type+'/data_test.json', accus, laws, label_type, all_accu_map, all_law_map)
    if data_pic is not None:
        train_df = train_df.sample(n=int(data_pic*train_df.shape[0]), axis=0)
        valid_df = valid_df.sample(n=int(data_pic*valid_df.shape[0]), axis=0)
        test_df = test_df.sample(n=int(data_pic*test_df.shape[0]), axis=0)
    return train_df, valid_df, test_df, accus, laws
    
    
    
    
    
#加载txt文件    
def load_arr_data(url):
    arr = []
    with open(url, encoding='utf-8') as arrfile:
        for line in arrfile:
            arr.append(str(line).strip())
    arr = process_arrdata(arr)
    return arr
    
    
#加载json文件    
def load_json_data(url, accus, laws, label_type, all_accu_map = None, all_law_map = None):
    train_x0 = [] # 案情描述，事实部分
    train_y01 = [] # imprisonment 刑期类型及长短
    train_y02 = [] # 被告人，背叛罪名，金钱惩罚，相关法律条文
    train_y11 = [] #accu对应编号
    train_y12 = [] #law对应编号
    train_y22 = []#刑期
    
    all_accu_arr = []
    all_law_arr = []
    fact_md5_arr = []
    with open(url, 'r', encoding='utf-8') as jsonfile:
        for line in jsonfile:
            item = json.loads(line)
            train_x0.append(item['fact'])    # 案情描述，事实部分 
            fact_md5 = hashlib.md5(item['fact'].encode("utf-8")).hexdigest()[8:-8]
            fact_md5_arr.append(fact_md5)
            imprisonment = item['meta'].pop('term_of_imprisonment') # imprisonment 刑期类型及长短
            if label_type == "single":
                train_y11.append(accus[item['meta']['accusation'][0]])
                train_y12.append(laws[str(item['meta']['relevant_articles'][0])])
            else:
                accr_arr = [accus[data] for data in item['meta']['accusation']]
                train_y11.append(accr_arr)
                law_arr = [laws[str(law)] for law in (item['meta'])['relevant_articles']]
                train_y12.append(law_arr)
            all_accu_arr += (item['meta']['accusation'])
            all_law_arr += (item['meta']['relevant_articles'])
            train_y22.append(gettime(imprisonment))
            train_y01.append(imprisonment)
            train_y02.append(item['meta']) # 被告人，背叛罪名，金钱惩罚，相关法律条文
    df = pd.concat([pd.DataFrame({"md5": fact_md5_arr}),pd.DataFrame({"fact": train_x0}),pd.DataFrame(train_y01),pd.DataFrame(train_y02),
                    pd.DataFrame({"accu": train_y11}),pd.DataFrame({"law": train_y12}),pd.DataFrame({"time": train_y22})], axis=1)
    df = df.reindex(np.random.permutation(df.index))
    if label_type == "single":
        if all_accu_map is None:
            all_accu_map = label_count(all_accu_arr)
        if all_law_map is None:
            all_law_map = label_count(all_law_arr)
        df["max_accu"] = get_max_label(df['accusation'].values, all_accu_map, accus)
        df["max_law"] = get_max_label(df['relevant_articles'].values, all_law_map, laws)
    return df, all_accu_map, all_law_map


#数组编号    
def process_arrdata(arr):
    datas = {}
    for i,value in enumerate(arr):
        datas[value] = i
    return datas


#刑期
def gettime(time):
    #将刑期用分类模型来做
    v = int(time['imprisonment'])

    if time['death_penalty']:
        return 0
    if time['life_imprisonment']:
        return 1
    elif v > 10 * 12:
        return 2
    elif v > 7 * 12:
        return 3
    elif v > 5 * 12:
        return 4
    elif v > 3 * 12:
        return 5
    elif v > 2 * 12:
        return 6
    elif v > 1 * 12:
        return 7
    else:
        return 8
    
    
#统计 每个label的次数   
def label_count(label_arr):
    label_map = {}
    for label in label_arr:
        if label in label_map:
            label_map[label] += 1
        else:
            label_map[label] = 1
    return label_map


#获取 最大值的label列表
def get_max_label(arr, label_map_count, label_map):
    label_max = []
    for item in arr:
        if len(item) == 1:
            max_label = item[0]
            label_max.append(label_map[str(max_label)])
        else:
            max_label = ""
            max_count = 0
            for data in item:
                if label_map_count[data] > max_count:
                    max_label = data
                    max_count = label_map_count[data]
            label_max.append(label_map[str(max_label)])
    return label_max


#将列表写入文件
def list_to_file(datas, path):
    with open(path, "w", encoding="utf-8") as f:
        for data in datas:
            line_text = ""
            if type(data) == list:
                for val in data:
                    line_text += val+"\t"
            else:
                line_text = data
            line_text += "\n"
            f.write(line_text)

#读取文件内容到列表
def file_to_list(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for text_line in f:
            text_line = str(text_line).rstrip("\n").strip()
            if "\t" in text_line:
                arr.append(text_line.split("\t"))
            else:
                arr.append(text_line)
    return arr



def write_to_file(arr, path):
    with open(path, "w", encoding="utf-8") as file:
        for data in arr:
            file.write(data)
            
            

def divide_file(path="./big_data/data_train.json"):
    import random
    arr = []
    with open(path, 'r', encoding='utf-8') as jsonfile:
        for line in jsonfile:
            arr.append(line)
        
    length = len(arr)
    random.shuffle(arr)
    test_arr = arr[:int(length*0.1)]
    valid_arr = arr[int(length*0.1): int(length*0.3)]
    train_arr = arr[int(length*0.3):]
    write_to_file(test_arr,"./big_data/data_test.json")
    write_to_file(valid_arr,"./big_data/data_valid.json")
    write_to_file(train_arr,"./big_data/data_train.json")
    return train_arr, valid_arr, test_arr
    
    
    
    
    
        
        
            
