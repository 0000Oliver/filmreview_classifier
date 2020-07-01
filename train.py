from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
import torch
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from zhconv import convert
import re
import os
import yaml
import easydict
import numpy as np
import random
import jieba #分词
from datapreprocess import build_word2id,build_word2vec,pre_weight,build_word_dict
from model import SentimentModel
from myutils import AvgrageMeter,accuracy,ConfuseMeter




def mycollate_fn(data):
    # 这里的data是getittem返回的（input，label）的二元组，总共有batch_size个
    data.sort(key=lambda x: len(x[0]), reverse=True)  # 根据input来排序
    data_length = [len(sq[0]) for sq in data]
    input_data = []
    label_data = []
    for i in data:
        input_data.append(i[0])
        label_data.append(i[1])
    input_data = pad_sequence(input_data, batch_first=True, padding_value=0)
    label_data = torch.tensor(label_data)
    return input_data, label_data, data_length

class CommentDataSet(Dataset):
    def __init__(self,data_path,word2ix):
        self.data_path = data_path
        self.word2ix = word2ix
        self.data ,self.label = self.get_data_label()
    def __getitem__(self, idx: int):
        return self.data[idx],self.label[idx]
    def __len__(self):
        return len(self.data)
    def get_data_label(self):
        data = []
        label =[]
        with open(self.data_path,'r',encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines :
                try :
                    label.append(torch.tensor(int(line[0]),dtype=torch.int64))
                except BaseException:
                    print('not expected line:' + line)
                    continue
                line = convert(line,'zh-cn')
                line_words = re.split(r"[\s]",line)[1:-1]
                words_to_idx = []
                for w in line_words:
                    try:
                        index =self.word2ix[w]
                    except BaseException:
                        index = 0
                    words_to_idx.append(index)
                data.append(torch.tensor(words_to_idx,dtype=torch.int64))
        return data, label

def train(configs):
    # 因为使用tensorboard画图会产生很多日志文件，这里进行清空操作
    import shutil
    if os.path.exists(configs.tensorboard_path):
        shutil.rmtree(configs.tensorboard_path)
        os.mkdir(configs.tensorboard_path)

    #word2ix,ix2word = build_word2id(configs.word2id_path)
    #word_vecs = build_word2vec(configs.pred_word2vec_path, word2ix, save_to_path=configs.word2vec_path)
    word2ix, ix2word, max_len, avg_len = build_word_dict(configs.train_path)

    word_vecs = pre_weight(word2ix,ix2word,configs)
    train_data = CommentDataSet(configs.train_path,word2ix)

    train_loader = DataLoader(train_data,batch_size=configs.batch_size,shuffle=configs.shuffle,
                              num_workers=configs.num_workers,collate_fn=mycollate_fn,)

    validation_data = CommentDataSet(configs.validation_path,word2ix)
    validation_loader = DataLoader(validation_data,batch_size=configs.batch_size,shuffle=configs.shuffle,
                                   num_workers=configs.num_workers,collate_fn=mycollate_fn,)
    test_data = CommentDataSet(configs.test_path,word2ix)
    test_loader = DataLoader(test_data,batch_size=configs.batch_size,shuffle=configs.shuffle,
                                   num_workers=configs.num_workers,collate_fn=mycollate_fn,)
    model = SentimentModel(embedding_dim=configs.embedding_dim,
                           hidden_dim=configs.hidden_dim,
                           pre_weight=word_vecs)
    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs.step_size, gamma=configs.lr_gamma)  # 学习率调整
    criterion = nn.CrossEntropyLoss()

    for epoch in range(configs.epochs):
        #模型训练
        model.train()
        top1 = AvgrageMeter()
        model = model.to(device)
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):# 0是下标起始位置默认为0
            inputs,labels ,batch_seq_len = data[0].to(device),data[1].to(device),data[2]

            #初始化为0，清楚上个拔头筹的梯度信息
            optimizer.zero_grad()

            outputs,hidden = model(inputs,batch_seq_len)

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            _,pred = outputs.topk(1)
            prec1 ,prec2 = accuracy(outputs,labels,topk=(1,2))
            n = inputs.size(0)
            top1.update(prec1.item(),n)
            train_loss += loss.item()
            postfix = {"epoch": epoch,'train_loss': '%.6f' %(train_loss/(i+1)),'train_acc':"%.6f" % top1.avg}
            tensorboard_path = configs.tensorboard_path
            if os.path.exists(tensorboard_path) ==False:
                os.mkdir(tensorboard_path)
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar("Train/Loss",loss.item(),epoch)
            writer.add_scalar("Train/Accuracy",top1.avg,epoch)
            writer.flush()

        print(postfix)

        #模型测试
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            val_top1 = AvgrageMeter()
            val_loss = 0.0
            for i , data in enumerate(validation_loader,0):
                inputs, labels, batch_seq_len = data[0].to(device), data[1].to(device), data[2]
                outputs,_= model(inputs,batch_seq_len)
                loss = criterion(outputs,labels)
                prec1,prec2 = accuracy(outputs,labels,topk=(1,2))
                n = inputs.size(0)
                val_top1.update(prec1.item(),n)
                val_loss+= loss.item()
                postfix={"validate_loss":"%.6f" % (val_loss/(i+1)),"validate_acc": '%.6f' % val_top1.avg}
                if os.path.exists(tensorboard_path) == False:
                    os.mkdir(tensorboard_path)
                writer = SummaryWriter(tensorboard_path)
                writer.add_scalar('Validate/Loss', loss.item(), epoch)
                writer.add_scalar('Validate/Accuracy', val_top1.avg, epoch)
                writer.flush()
            #模型保存
            if val_top1.avg>val_acc:
                if os.path.exists(configs.model_save_path)==False:
                    os.mkdir("./modelDict/")
                torch.save(model.state_dict(),configs.model_save_path)
            val_acc = val_top1.avg
            print(postfix)

        scheduler.step()
    confuse_meter =test(test_loader,device,model,criterion)
    print('prec:%.6f  recall:%.6f  F1:%.6f' % (confuse_meter.pre, confuse_meter.rec, confuse_meter.F1))
    print(confuse_meter.confuse_mat)

def test(validate_loader, device, model, criterion):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    confuse_meter = ConfuseMeter()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = validate_loader
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels, batch_seq_len = data[0].to(device), data[1].to(device), data[2]
            #         inputs,labels = data[0],data[1]
            outputs,_ = model(inputs, batch_seq_len)
#             loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            confuse_meter.update(outputs, labels)
#             validate_loss += loss.item()
            postfix = { 'test_acc': '%.6f' % val_top1.avg,
                      'confuse_acc': '%.6f' % confuse_meter.acc}
           # validate_loader.set_postfix(log=postfix)
            print(postfix)
        val_acc = val_top1.avg
    return confuse_meter
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def predict(comment_str, model, device):
    model = model.to(device)
    seg_list = jieba.lcut(comment_str,cut_all=False)
    words_to_idx = []
    for w in seg_list:
        try:
            index = word2ix[w]
        except:
            index = 0 #可能出现没有收录的词语，置为0
        words_to_idx.append(index)
    inputs = torch.tensor(words_to_idx).to(device)
    inputs = inputs.reshape(1,len(inputs))
    outputs,_ = model(inputs, [len(inputs),])
    pred = outputs.argmax(1).item()
    return pred











if __name__ =="__main__":
    with open("./Configs.yaml", 'r') as f:

        configs = yaml.safe_load(f)
        configs = easydict.EasyDict(configs)
        set_seed(configs.seed)
        #train(configs)
        word2ix, ix2word, max_len, avg_len = build_word_dict(configs.train_path)
        word_vecs = pre_weight(word2ix, ix2word, configs)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = SentimentModel(embedding_dim=configs.embedding_dim,
                               hidden_dim=configs.hidden_dim,
                               pre_weight=word_vecs)
        model.load_state_dict(torch.load(configs.model_save_path,map_location=torch.device('cpu')), strict=True)  # 模型加载
        model.to(device)



        #加勒比海盗影评
        comment_str1 = "又臭又长，漏洞百出，毫无逻辑，结局脑残。海盗该表现出来的气势激斗完全不到位"

        if (predict(comment_str1, model, device)):
            print(comment_str1)
            print("Negative")
        else:
            print(comment_str1)
            print("Positive")

        comment_str2 ="加勒比海盗这一系列我都特别喜欢，喜欢杰克幽默风趣，顽强，坚韧勇敢，像永远也打不死的小强，还有他的黑珍珠海盗船，太神奇了！"
        print(comment_str2)
        if (predict(comment_str2, model, device)):
            print("Negative")
        else:
            print("Positive")




