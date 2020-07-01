import torch
class AvgrageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self,val,n=1):
        self.sum += val*n
        self.cnt +=n
        self.avg = self.sum /self.cnt

class ConfuseMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.confuse_mat = torch.zeros(2,2)
        self.tp = self.confuse_mat[0,0]
        self.fp = self.confuse_mat[0,1]
        self.tn = self.confuse_mat[1,1]
        self.fn = self.confuse_mat[1,0]
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.F1 = 0
        self.acc =(self.tp + self.tn) /self.confuse_mat.sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.F1 = 2* self.pre*self.rec / (self.pre+ self.rec)
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
def accuracy(output ,label,topk =(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(label.view(1,-1).expand_as(pred))

    rtn =[]
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        rtn.append(correct_k.mul_(100.0 / batch_size))
    return rtn
