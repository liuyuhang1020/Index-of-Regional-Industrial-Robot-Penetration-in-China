import numpy as np
import pandas as pd
import logging
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from gensim.models import doc2vec
from mittens import GloVe
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

vectorizer=CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform([" ".join(i) for i in data["经营范围分词"]])) 
x_tfidf = tfidf.toarray()

sentences = list(data["经营范围分词"])
## 模型训练，生成词向量
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 训练skip-gram模型; 小于2的单词被剔除，默认window=5
model_word2vec = word2vec.Word2Vec(sentences, vector_size=100, window=5, 
                                   min_count=4, workers=4, alpha=0.025,
                                   min_alpha=0.025, sg=1,) 

## 获取word2vec训练出的每条新闻词向量的和及平均值，记为x_sum和x_mean
def compute_doc_vec_sum(news):
    vec = np.zeros(model_word2vec.layer1_size, dtype=np.float32)
    for word in news:
        if word in model_word2vec.wv.key_to_index:
            vec += model_word2vec.wv[word]  # 求所有词向量的和
    return vec  # 求和

def compute_doc_sum(news):
    return np.row_stack([compute_doc_vec_sum(x) for x in news])

def compute_doc_vec_mean(news):
    vec = np.zeros(model_word2vec.layer1_size, dtype=np.float32)
    n = 0
    for word in news:
        if word in model_word2vec.wv.key_to_index:
            vec += model_word2vec.wv[word]  # 求所有词向量的和
            n += 1  # 计算词的个数
    return vec / n  # 求平均值

def compute_doc_mean(news):
    return np.row_stack([compute_doc_vec_mean(x) for x in news])
    
x_word2vec_sum = compute_doc_sum(data["经营范围分词"])
x_word2vec_mean = compute_doc_mean(data["经营范围分词"])

def X_train(cut_sentence):
    x_train = []
    for i, sentence in enumerate(cut_sentence):
        document = doc2vec.TaggedDocument(sentence,tags=[i])
        x_train.append(document)
    return x_train

documents = X_train(sentences)
model_doc2vec = doc2vec.Doc2Vec(documents, vector_size=100, window=5, 
                                min_count=4, workers=4, alpha=0.025,
                                min_alpha=0.025, dm=0)
                     
x_docvec = np.row_stack([model_doc2vec.docvecs[i]] for i in range(len(documents)))


## 计算词频

def word_count(sentences):
    word_freq = collections.defaultdict(int)
    for sentence in sentences:  
        for w in sentence:
            word_freq[w] += 1
    return word_freq
    #return word_freq.items()   该语句返回值的类型为list

word_freq = word_count(sentences)
# 筛选出词频大于等于4的词
word_freq_big = {k:v for k, v in word_freq.items() if v >= 4}
# 按照词频进行降序排列
word_freq_sorted = sorted(word_freq_big.items(), key=lambda x:x[1], reverse=True)
df = pd.DataFrame(word_freq_sorted, columns=['Word', 'Frequency'])
df.set_index(['Word'],inplace=True)  # 将词语列设为index

## 按照词频由高到低构建words_dict，对单词进行编码
def build_dict(sentences, min_word_freq=4):
    word_freq = word_count(sentences) 
    word_freq = filter(lambda x: x[1] >= min_word_freq, word_freq.items()) 
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    # key用于指定排序的元素，因为sorted默认使用list中每个item的第一个元素从小到
    #大排列，所以这里通过lambda进行前后元素调序，并对词频去相反数，从而将词频最大的排列在最前面
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, range(len(words))))
    return word_idx

words_dict = build_dict(sentences, 2)

## 将分词后的结果转为词对应的编码
text_clean_split_id = []
for text in data["经营范围分词"]:
    idx_ = []
    for word in text:
        if word in words_dict.keys():
            idx = str(words_dict[word])
            idx_.append(idx)
    text_clean_split_id.append(idx_)
data["经营范围分词编码"] = text_clean_split_id

## 计算共现矩阵
def countCOOC(cooccurrence, window, coreIndex):
# cooccurrence：当前共现矩阵
# window：当前移动窗口数组
# coreIndex：当前移动窗口数组中的窗口中心位置
    for index in range(len(window)):
        if index == coreIndex:
            continue
        else:
            cooccurrence[window[coreIndex]][window[index]] = cooccurrence[window[coreIndex]][window[index]] + 1
    return cooccurrence

coWindow = 5 # 共现窗口大小（半径）
tableSize = len(words_dict) # 共现矩阵维度
cooccurrence = np.zeros((tableSize, tableSize), "int64" )
for item in data["经营范围分词编码"]:
    itemInt = [int(x) for x in item]
    for core in range(len(itemInt)):
        if core < coWindow:
            # 左窗口不足
            window = itemInt[0: core + coWindow +1]
            coreIndex = core
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)
        elif core > len(item) - 1 - coWindow:
            # 右窗口不足
            window = itemInt[core - coWindow:(len(item))]
            coreIndex = coWindow
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)
        else:
        # 左右均没有问题
            window = itemInt[core - coWindow: core + coWindow + 1]
            coreIndex = coWindow
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

## 模型训练，生成词向量

vecLength=100           # 矩阵长度
max_iter=500         # 最大迭代次数
display_progress=5   # 每次展示
glove_model = GloVe(n=vecLength, max_iter=max_iter, display_progress=display_progress)
# 模型训练
embeddings_100 = glove_model.fit(cooccurrence)
df = pd.DataFrame(data = embeddings_100, index = words_dict.keys())

# 存储词向量结果
def gen_glove(filename, embeddings):
    df = pd.DataFrame(data = embeddings_100, index = words_dict.keys())
    df.to_csv(filename, sep='\t', header = False)

gen_glove('glove_100.txt', embeddings_100)

glove_vec = pd.read_csv('glove_100.txt', sep="\t", header=None)
glove_vec.set_index([0],inplace=True)

## 获取glove训练出的每条新闻词向量的和及平均值，记为x_freq_sum和x_freq_mean
vecLength = 100

vec_listt_sum = []
for text in data["经营范围分词"]:
    vec = np.zeros(vecLength, dtype=np.float32)
    vec_list = []
    for word in text:
        if word in glove_vec.index:
            vec += glove_vec.loc[word]
    vec_list.append(vec)
    vec_listt_sum.append(vec_list)
x_glove_sum = np.array(vec_listt_sum)

x_glove_sum = x_glove_sum.reshape(-1, vecLength)

vec_listt_mean = []
for text in data["经营范围分词"]:
    vec = np.zeros(vecLength, dtype=np.float32)
    vec_list = []
    n = 0
    for word in text:
        if word in glove_vec.index:
            vec += glove_vec.loc[word]
            n += 1
    vec_list.append(vec / n)
    vec_listt_mean.append(vec_list)
x_glove_mean = np.array(vec_listt_mean)
x_glove_mean = x_glove_mean.reshape(-1, vecLength)

x_total = np.concatenate((x_word2vec_sum, x_word2vec_mean, x_docvec, x_glove_sum, x_glove_mean), axis=1)
x_word2vec = np.concatenate((x_word2vec_sum, x_word2vec_mean), axis=1)
x_glove = np.concatenate((x_glove_sum, x_glove_mean), axis=1)

Xs = [x_total, x_tfidf, x_word2vec, x_word2vec_sum, x_word2vec_mean, x_docvec, x_glove, x_glove_sum, x_glove_mean]

clfs = [GaussianNB(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(n_jobs=-1), GradientBoostingClassifier(), XGBClassifier(), LGBMClassifier(), MLPClassifier()]
samplings = [None, RandomOverSampler(), SMOTE(), ADASYN(), RandomUnderSampler(), SMOTEENN(), SMOTETomek()]

clf_names = ["GNB", "LR", "DT", "RF", "GB", "XGB", "LGBM", "MLP"]
X_names = {1: "total", 2: "tfidf", 3: "word2vec", 4: "word2vec_sum", 5: "word2vec_mean", 6: "docvec", 7: "glove", 8: "glove_sum", 9: "glove_mean"}
sampling_names = {1: "None", 2: "RandomOver", 3: "SMOTE", 4: "ADASYN", 5: "RandomUnder", 6: "SMOTEENN", 7: "SMOTETomek"}

index = []
for X_name in X_names.keys():
    for sampling_name in sampling_names.keys():
        index.append(10*X_name + sampling_name)

cmat_df = pd.DataFrame(columns=clf_names, index=index)

def split_train_test(x, y, test_size):
    train_idx, test_idx = train_test_split(range(len(y)), test_size=test_size, stratify=y)
    x_train = x[train_idx, :]
    y_train = y[train_idx]
    x_test = x[test_idx, :]
    y_test = y[test_idx]
    return x_train, y_train, x_test, y_test

def eval(clf, X, sampling=None, test_size=0.3):
    x_train, y_train, x_test, y_test = split_train_test(X, y, test_size)
    if sampling:
        x_resampled, y_resampled = sampling.fit_resample(x_train, y_train)
    else:
        x_resampled, y_resampled = x_train, y_train
    clf.fit(x_resampled, y_resampled)
    y_pred = clf.predict(x_test)
    return confusion_matrix(y_test, y_pred)

for i, clf in enumerate(clfs):
    for a, X in enumerate(Xs):
        for b, sampling in enumerate(samplings):
            cmat_df.loc[10*(a + 1) + b + 1, clf_names[i]] = eval(clf, X, sampling)
