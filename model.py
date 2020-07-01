import torch
# 变长序列的处理
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch import nn
import yaml
import easydict
f = open("./Configs.yaml", 'r')
Config = yaml.safe_load(f)
f.close()
Config = easydict.EasyDict(Config)
class SentimentModel(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,pre_weight):
        super(SentimentModel,self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(pre_weight)

        self.embeddings.weight.requires_grad =True
        self.lstm = nn.LSTM(embedding_dim,self.hidden_dim,num_layers=Config.LSTM_layers,
                            batch_first=Config.batch_first,dropout=Config.drop_prob,bidirectional=Config.bidirectional)
        self.dropout = nn.Dropout(Config.drop_prob)
        self.fc1 = nn.Linear(self.hidden_dim,256)
        self.fc2 = nn.Linear(256,32)
        self.fc3 = nn.Linear(32,2)
    def forward(self,input,batch_seq_len,hidden= None):
        embeds = self.embeddings(input)
        embeds = pack_padded_sequence(embeds,batch_seq_len,batch_first=True)
        batch_size,seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(Config.LSTM_layers*1,batch_size,self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.LSTM_layers*1,batch_size,self.hidden_dim).fill_(0).float()
        else:
            h_0 ,c_0 = hidden
        output,hidden = self.lstm(embeds,(h_0,c_0))
        output,_ = pad_packed_sequence(output,batch_first=True)
        output = self.dropout(torch.tanh(self.fc1(output)))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        last_output = self.get_last_output(output,batch_seq_len)

        return last_output,hidden
    def get_last_output(self,output,batch_seq_len):
        last_outputs = torch.zeros((output.shape[0],output.shape[2]))
        for i in range(len(batch_seq_len)):
            last_outputs[i] = output[i][batch_seq_len[i]-1]
        last_outputs =last_outputs.to(output.device)
        return last_outputs
# class SentimentModel(nn.Module):
#     def __init__(self,embedding_dim,hidden_dim,pre_weight):
#         super(SentimentModel,self).__init__()
#         self.hidden_dim = hidden_dim
#         self.embeddings = nn.Embedding.from_pretrained(pre_weight)
#
#         self.embeddings.weight.requires_grad =True
#         self.lstm = nn.LSTM(embedding_dim,self.hidden_dim,num_layers=Config.LSTM_layers,
#                             batch_first=Config.batch_first,dropout=Config.drop_prob,bidirectional=Config.bidirectional)
#         self.dropout = nn.Dropout(Config.drop_prob)
#         self.fc1 = nn.Linear(self.hidden_dim,256)
#         self.fc2 = nn.Linear(256,32)
#         self.fc3 = nn.Linear(32,2)
# #         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，
#     def forard(self, input, batch_seq_len, hidden=None):
#         embeds = self.embeddings(input)
#         embeds = pack_padded_sequence(embeds, batch_seq_len, batch_first=True)
#         batch_size, seq_len = input.size()
#         if hidden is None:
#             h_0 = input.data.new(Config.LSTM_Layres * 1, batch_size, self.hidden_dim).fill_(0).float()
#             c_0 = input.data.new(Config.LSTM_Layres * 1, batch_size, self.hidden_dim).fill_(0).float()
#         else:
#             h_0, c_0 = hidden
#         output, hidden = self.lstm(embeds, (h_0, c_0))
#         output, _ = pad_packed_sequence(output, batch_first=True)
#         output = self.dropout(torch.tanh(self.fc1(output)))
#         output = torch.tanh(self.fc2(output))
#         output = self.fc3(output)
#         last_output = self.get_last_output(output, batch_seq_len)
#
#         return last_output, hidden
#     def get_last_output(self,output,batch_seq_len):
#         last_outputs = torch.zeros((output.shape[0],output.shape[2]))
#         for i in range(len(batch_seq_len)):
#             last_outputs[i] =  output[i][batch_seq_len[i]-1]#index 是长度 -1
#         last_outputs = last_outputs.to(output.device)
#         return last_outputs
#
#
#
