import pandas
import re
from tqdm import tqdm
from preprocess import preprocess
from standardize_jieba import standardize

data_path="C:\\Users\\86191\\Desktop\\sample_220325_3.CSV"
id_list=list(pandas.read_csv(data_path,encoding='gbk')['t1_esiecid'])
opscope_list=list(pandas.read_csv(data_path,encoding='gbk')['t2_business_tyc'])
root_path="C:\\Users\\86191\\OneDrive\\CodeField\\CODE_PYTHON\\PYTHON_Project\\企业经营范围文本向量化\\词库\\标准化词根.csv"
root=pandas.read_csv(root_path,encoding='gbk')
phrase_path="C:\\Users\\86191\\OneDrive\\CodeField\\CODE_PYTHON\\PYTHON_Project\\企业经营范围文本向量化\\词库\\标准化词组.csv"
phrase=pandas.read_csv(phrase_path,encoding='gbk')
corpus=[]
for i in tqdm(range(len(opscope_list))):
    id=id_list[i]
    opscope=opscope_list[i]
    sentence_list=preprocess(str(opscope))
    standard_list=[]
    for sentence in sentence_list:
        standard=standardize(sentence,root,phrase)
        if standard:
            standard_list.extend(standard)
        else:
            standard_list.extend(re.split('[、 ：]',sentence))
    if standard_list!=[]:
        corpus.append([id,' '.join(standard_list)])
corpus=pandas.DataFrame(corpus,columns=['t1_esiecid','t2_business_tyc'])
save_path="C:\\Users\\86191\\Desktop\\sample_220325_3_standard.CSV"
corpus.to_csv(save_path,encoding='gbk')