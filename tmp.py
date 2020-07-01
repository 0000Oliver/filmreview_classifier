import yaml
import easydict
from zhconv import convert #简繁转换
import re
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
import torch
from torch.utils.data import Dataset,DataLoader
import gensim
from torch import nn,optim
import os
from torch.utils.tensorboard import SummaryWriter
#from tqdm.notebook import tqdm
import numpy as np
import random
from model import SentimentModel
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
    def __init__(self, data_path, word2ix, ix2word):
        self.data_path = data_path
        self.word2ix = word2ix
        self.ix2word = ix2word
        self.data, self.label = self.get_data_label()

    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

    def get_data_label(self):
        data = []
        label = []
        with open(self.data_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    label.append(torch.tensor(int(line[0]), dtype=torch.int64))
                except BaseException:  # 遇到首个字符不是标签的就跳过比如空行，并打印
                    print('not expected line:' + line)
                    continue
                line = convert(line, 'zh-cn')  # 转换成大陆简体
                line_words = re.split(r'[\s]', line)[1:-1]  # 按照空字符\t\n 空格来切分
                words_to_idx = []
                for w in line_words:
                    try:
                        index = self.word2ix[w]
                    except BaseException:
                        index = 0  # 测试集，验证集中可能出现没有收录的词语，置为0
                    #                 words_to_idx = [self.word2ix[w] for w in line_words]
                    words_to_idx.append(index)
                data.append(torch.tensor(words_to_idx, dtype=torch.int64))
        return data, label

# class SentimentModel(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim,pre_weight):
#         super(SentimentModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.embeddings = nn.Embedding.from_pretrained(pre_weight)
#         # requires_grad指定是否在训练过程中对词向量的权重进行微调
#         self.embeddings.weight.requires_grad = True
#         self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=configs.LSTM_layers,
#                             batch_first=True, dropout=configs.drop_prob, bidirectional=False)
#         self.dropout = nn.Dropout(configs.drop_prob)
#         self.fc1 = nn.Linear(self.hidden_dim,256)
#         self.fc2 = nn.Linear(256,32)
#         self.fc3 = nn.Linear(32,2)
# #         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，
#
#     def forward(self, input, batch_seq_len, hidden=None):
#         embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
#         embeds = pack_padded_sequence(embeds,batch_seq_len, batch_first=True)
#         batch_size, seq_len = input.size()
#         if hidden is None:
#             h_0 = input.data.new(configs.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
#             c_0 = input.data.new(configs.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
#         else:
#             h_0, c_0 = hidden
#         output, hidden = self.lstm(embeds, (h_0, c_0))#hidden 是h,和c 这两个隐状态
#         output,_ = pad_packed_sequence(output,batch_first=True)
#
#         output = self.dropout(torch.tanh(self.fc1(output)))
#         output = torch.tanh(self.fc2(output))
#         output = self.fc3(output)
#         last_outputs = self.get_last_output(output, batch_seq_len)
#         #         output = output.reshape(batch_size * seq_len, -1)
#         return last_outputs, hidden
#
#     def get_last_output(self, output, batch_seq_len):
#         last_outputs = torch.zeros((output.shape[0], output.shape[2]))
#         for i in range(len(batch_seq_len)):
#             last_outputs[i] = output[i][batch_seq_len[i] - 1]  # index 是长度 -1
#         last_outputs = last_outputs.to(output.device)
#         return last_outputs


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
class ConfuseMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        # 标签的分类：0 pos 1 neg
        self.confuse_mat = torch.zeros(2,2)
        self.tp = self.confuse_mat[0,0]
        self.fp = self.confuse_mat[0,1]
        self.tn = self.confuse_mat[1,1]
        self.fn = self.confuse_mat[1,0]
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.F1 = 0
    def update(self, output, label):
        pred = output.argmax(dim = 1)
        for l, p in zip(label.view(-1),pred.view(-1)):
            self.confuse_mat[p.long(), l.long()] += 1 # 对应的格子加1
        self.tp = self.confuse_mat[0,0]
        self.fp = self.confuse_mat[0,1]
        self.tn = self.confuse_mat[1,1]
        self.fn = self.confuse_mat[1,0]
        self.acc = (self.tp+self.tn) / self.confuse_mat.sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.F1 = 2 * self.pre*self.rec / (self.pre + self.rec)

## topk的准确率计算
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)
    #print(output)
   # print(label)
    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True)
   # print(pred)#使用topk来获得前k个的索引
    pred = pred.t() # 进行转置
    #print(pred)
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred)) # 与正确标签序列形成的矩阵相比，生成True/False矩阵
#     print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size)) # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn

#一个epoch的训练逻辑
def train(epoch,epochs, train_loader, device, model, criterion, optimizer,scheduler,tensorboard_path):
    model.train()
    top1 = AvgrageMeter()
    model = model.to(device)
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
        inputs, labels, batch_seq_len = data[0].to(device), data[1].to(device), data[2]
        # 初始为0，清除上个batch的梯度信息
        optimizer.zero_grad()
        outputs,hidden = model(inputs,batch_seq_len)

        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        _,pred = outputs.topk(1)
        prec1, prec2= accuracy(outputs, labels, topk=(1,2))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
        # train_loader.set_postfix(log=postfix)
        print('epoch:',epoch)
        print(postfix)
        # ternsorboard 曲线绘制
        if os.path.exists(tensorboard_path) == False:
            os.mkdir(tensorboard_path)
        writer = SummaryWriter(tensorboard_path)
        writer.add_scalar('Train/Loss', loss.item(), epoch)
        writer.add_scalar('Train/Accuracy', top1.avg, epoch)
        writer.flush()
    scheduler.step()

    #     print('Finished Training')

def validate(epoch,validate_loader, device, model, criterion, tensorboard_path):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = validate_loader#tqdm()
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels, batch_seq_len = data[0].to(device), data[1].to(device), data[2]
            #         inputs,labels = data[0],data[1]
            outputs,_ = model(inputs, batch_seq_len)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validate_loss': '%.6f' % (validate_loss / (i + 1)), 'validate_acc': '%.6f' % val_top1.avg}
            validate_loader.set_postfix(log=postfix)

            # ternsorboard 曲线绘制
            if os.path.exists(tensorboard_path) == False:
                os.mkdir(tensorboard_path)
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Validate/Loss', loss.item(), epoch)
            writer.add_scalar('Validate/Accuracy', val_top1.avg, epoch)
            writer.flush()
        val_acc = val_top1.avg
    return val_acc
#混淆矩阵指标
def validate(epoch,validate_loader, device, model, criterion, tensorboard_path):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = validate_loader#tqdm()
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels, batch_seq_len = data[0].to(device), data[1].to(device), data[2]
            #         inputs,labels = data[0],data[1]
            outputs,_ = model(inputs, batch_seq_len)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validate_loss': '%.6f' % (validate_loss / (i + 1)), 'validate_acc': '%.6f' % val_top1.avg}
            print(postfix)
            #validate_loader.set_postfix(log=postfix)

            # ternsorboard 曲线绘制
            if os.path.exists(tensorboard_path) == False:
                os.mkdir(tensorboard_path)
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Validate/Loss', loss.item(), epoch)
            writer.add_scalar('Validate/Accuracy', val_top1.avg, epoch)
            writer.flush()
        val_acc = val_top1.avg
    return val_acc
def test(validate_loader, device, model, criterion):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    confuse_meter = ConfuseMeter()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = validate_loader#tqdm()
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
            print(postfix)
            #validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return confuse_meter

def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  #并行gpu
        torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
#         torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速


if __name__ == "__main__":
    with open("./Configs.yaml",'r') as f:
        configs = yaml.safe_load(f)
        configs = easydict.EasyDict(configs)
        print(configs)
        word2ix, ix2word, max_len, avg_len = build_word_dict(configs.train_path)
        # print(word2ix)
        # print(max_len, avg_len)
        train_data = CommentDataSet(configs.train_path, word2ix, ix2word)
        # data ,label = train_data
        # print(data)
        # print(label)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                                  num_workers=0, collate_fn=mycollate_fn, )
        # for (input_data, label_data, data_length) in train_loader:
        #     print(input_data)
        #     print(label_data)
        #     print(data_length)

        validation_data = CommentDataSet(configs.validation_path, word2ix, ix2word)
        validation_loader = DataLoader(validation_data, batch_size=16, shuffle=True,
                                       num_workers=0, collate_fn=mycollate_fn, )

        test_data = CommentDataSet(configs.test_path, word2ix, ix2word)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False,
                                 num_workers=0, collate_fn=mycollate_fn, )
        # word2vec加载
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(configs.pred_word2vec_path, binary=True)
        # 50维的向量
        #print(word2vec_model.__dict__['vectors'].shape)
        # (426677, 50)


        def pre_weight(vocab_size):
            weight = torch.zeros(vocab_size, configs.embedding_dim)
            # 初始权重
            for i in range(len(word2vec_model.index2word)):  # 预训练中没有word2ix，所以只能用索引来遍历
                try:
                    index = word2ix[word2vec_model.index2word[i]]  # 得到预训练中的词汇的新索引
                except:
                    continue
                weight[index, :] = torch.from_numpy(word2vec_model.get_vector(
                    ix2word[word2ix[word2vec_model.index2word[i]]]))  # 得到对应的词向量
            return weight


        # pre_weight = pre_weight(len(word2ix))
        # print(pre_weight.shape)
        #torch.Size([51406, 50])
        set_seed(configs.seed)
        model = SentimentModel(embedding_dim=configs.embedding_dim,
                               hidden_dim=configs.hidden_dim,
                               pre_weight=pre_weight(len(word2ix)))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epochs = 10
        optimizer = optim.Adam(model.parameters(), lr=configs.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调整
        criterion = nn.CrossEntropyLoss()

        import shutil

        if os.path.exists(configs.tensorboard_path):
            shutil.rmtree(configs.tensorboard_path)
            os.mkdir(configs.tensorboard_path)

        for epoch in range(epochs):
            print(len(train_loader))
            train_loader1 = train_loader#tqdm
            # train_loader1.set_description(
            #     '[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))
            train(epoch, epochs, train_loader1, device, model, criterion, optimizer, scheduler, configs.tensorboard_path)
            validate(epoch, validation_loader, device, model, criterion, configs
                     .tensorboard_path)
        # 模型保存
        if os.path.exists(configs.model_save_path) == False:
            os.mkdir('./modelDict/')
        torch.save(model.state_dict(), configs.model_save_path)
        model_test = SentimentModel(embedding_dim=configs.embedding_dim,
                                    hidden_dim=configs.hidden_dim,
                                    pre_weight=pre_weight(len(word2ix)))
        optimizer_test = optim.Adam(model_test.parameters(), lr=configs.lr)
        scheduler_test = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调整
        criterion_test = nn.CrossEntropyLoss()
        model_test.load_state_dict(torch.load(configs.model_save_path), strict=True)  # 模型加载

        confuse_meter = ConfuseMeter()
        confuse_meter = test(test_loader, device, model_test, criterion_test)
        print('prec:%.6f  recall:%.6f  F1:%.6f' % (confuse_meter.pre, confuse_meter.rec, confuse_meter.F1))
        print(confuse_meter.confuse_mat)





