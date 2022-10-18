import pandas
import re
from tqdm import tqdm
from preprocess import preprocess
from standardize_jieba import standardize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

data_path = "C:\\Users\\86191\\Desktop\\sample_220325_3.CSV"
id_list = list(pandas.read_csv(data_path, encoding='gbk')['t1_esiecid'])
opscope_list = list(pandas.read_csv(data_path, encoding='gbk')['t2_business_tyc'])
root_path = "C:\\Users\\86191\\OneDrive\\CodeField\\CODE_PYTHON\\PYTHON_Project\\企业经营范围文本向量化\\词库\\标准化词根.csv"
root = pandas.read_csv(root_path, encoding='gbk')
phrase_path = "C:\\Users\\86191\\OneDrive\\CodeField\\CODE_PYTHON\\PYTHON_Project\\企业经营范围文本向量化\\词库\\标准化词组.csv"
phrase = pandas.read_csv(phrase_path, encoding='gbk')
corpus = []
for i in tqdm(range(len(opscope_list))):
    id = id_list[i]
    opscope = opscope_list[i]
    sentence_list = preprocess(str(opscope))
    standard_list = []
    for sentence in sentence_list:
        standard = standardize(sentence, root, phrase)
        if standard:
            standard_list.extend(standard)
        else:
            standard_list.extend(re.split('[、 ：]', sentence))
    if standard_list != []:
        corpus.append([id, ' '.join(standard_list)])
corpus = pandas.DataFrame(corpus,columns=['t1_esiecid','t2_business_tyc'])
save_path = "C:\\Users\\86191\\Desktop\\sample_220325_3_standard.CSV"
corpus.to_csv(save_path, encoding='gbk')

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
    
x_word2vec_sum = compute_doc_sum(corpus)
x_word2vec_mean = compute_doc_mean(corpus)

param_dist = {'n_estimators': 1000,
              'eta': 0.01,  # 学习率，通过减少每一步的权重，可以提高模型的稳健性，0.01-0.2
              'alpha': 1,  # 惩罚项参数：γ
              'lambda': 0.001,  # 惩罚项参数：λ
              }

x_train, y_train, x_test, y_test = split_train_test(X, y, test_size)
x_resampled, y_resampled = SMOTE.fit_resample(x_train, y_train)
clf = XGBClassifier(**param_dist)
clf.fit(x_resampled, y_resampled)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4), dpi=100)
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(cmat, cmap=cmap, annot=True, fmt='.20g')
plt.ylabel('Actual label')
plt.xlabel('Predict label')
plt.show()
