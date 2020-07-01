import numpy as np
import yaml
import easydict
import gensim
import torch
from zhconv import convert #简繁转换
import re

def build_word2id(file):
     """
    :param file: word2id保存地址
    :return: None
    """
     word2id = {'_PAD_':0}
     id2word = {0:'_PAD_'}
     path= ['./data/train.txt','./data/validation.txt','./data/test.txt']
     for _path in path:
         with open(_path,encoding='utf-8') as f:
             for line in f.readlines():
                 sp = line.strip().split()
                 for word in sp[1:]:
                     if word not in word2id.keys():

                         word2id[word] = len((word2id))
                         id2word[len((word2id))] = word
     with open(file,'w',encoding='utf-8') as f:
         for w in word2id:
             f.write(w+'\t')
             f.write(str(word2id[w]))
             f.write('\n')
     return word2id,id2word

def build_word2vec(word2vec,word2id,save_to_path = None):
    """

    :param fname: 预训练的word2vec
    :param word2id: 预料文本中包含的词汇集
    :param save_to_path: 保存训练语料库中词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec 向量{id:word2vec}.
    """


    n_words = max(word2id.values())+1
    test =1
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec,binary=True)

    word_vecs = np.array(np.random.uniform(-1,1,[n_words,model.vector_size]))
    for word in word2id.keys():

        try:
            word_vecs[word2id[word]] = model.get_vector(word)
        except KeyError:
            pass
        if save_to_path:
            with open(save_to_path,'w',encoding='utf-8') as f:
                for vec in word_vecs:
                    vec = [str(w) for w in vec]
                    f.write(''.join(vec))
                    f.write("\n")
        return  torch.from_numpy(word_vecs)
# 简繁转换 并构建词汇表
def build_word_dict(train_path):
    words = []
    max_len = 0
    total_len = 0
    with open(train_path,'r',encoding='UTF-8') as f:
        lines = f.readlines()
        for line in  lines:
            line = convert(line, 'zh-cn') #转换成大陆简体
            line_words = re.split(r'[\s]', line)[1:-1] # 按照空字符\t\n 空格来切分
            max_len = max(max_len, len(line_words))
            total_len += len(line_words)
            for w in line_words:
                words.append(w)
    words = list(set(words))#最终去重
    words = sorted(words) # 一定要排序不然每次读取后生成此表都不一致，主要是set后顺序不同
    #用unknown来表示不在训练语料中的词汇
    word2ix = {w:i+1 for i,w in enumerate(words)} # 第0是unknown的 所以i+1
    ix2word = {i+1:w for i,w in enumerate(words)}
    word2ix['<unk>'] = 0
    ix2word[0] = '<unk>'
    avg_len = total_len / len(lines)
    return word2ix, ix2word, max_len,  avg_len
def pre_weight(word2ix,ix2word,Config):
    # word2vec加载
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(Config.pred_word2vec_path, binary=True)
    # 50维的向量
    weight = torch.zeros(len(word2ix),Config.embedding_dim)
    #初始权重
    for i in range(len(word2vec_model.index2word)):#预训练中没有word2ix，所以只能用索引来遍历
        try:
            index = word2ix[word2vec_model.index2word[i]]#得到预训练中的词汇的新索引
        except:
            continue
        weight[index, :] = torch.from_numpy(word2vec_model.get_vector(
            ix2word[word2ix[word2vec_model.index2word[i]]]))#得到对应的词向量
    return weight

if __name__ == "__main__":
    with open("./Configs.yaml",'r') as f:
        configs = yaml.safe_load(f)
        configs = easydict.EasyDict(configs)
        print(configs)
        word2id =build_word2id(configs.word2id_path)
        print(len(word2id))
        word_vecs=build_word2vec(configs.pred_word2vec_path,word2id,save_to_path =configs.word2vec_path)
        print(len(word_vecs))
        print(len(word_vecs[0]))
        # 59290
        # 50



