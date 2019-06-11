import torch
import torch.nn as nn
from torch.utils import data as Data
import numpy as np
import os
import pickle
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # (batch,2,563,59)
            torch.nn.Conv2d(
                in_channels=2,
                out_channels=8,
                kernel_size=5,
                stride=2,
                padding=0),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8),
            nn.Dropout(0.2),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, (2,1), 1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            # nn.Dropout(0.2),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, (2,1), 1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.2),
            nn.Tanh()
        )
        self.gru = nn.GRU(32,16,2,batch_first=True)
        self.lin = nn.Sequential(
            nn.Linear(2*8*16, 128),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.contiguous()
        # print(x.shape)
        x = x.view(x.size(0),32,-1)
        x = x.transpose(1,2)
        # print(x.shape)
        x,h = self.gru(x)
        x = x.contiguous()
        x = x.view(x.size(0),-1)
        x = self.lin(x)
        return x


def get_data():
    f = open('/home/asr4/sea/dcase/workarea/DSS/traindata_9_t_1.pkl','rb')
    # f1 = abs(pickle.load(f))
    f1 = abs(pickle.load(f))
    f.close()
    print('load 1')
    f = open('/home/asr4/sea/dcase/workarea/DSS/traindata_9_t_2.pkl','rb')
    f2 = abs(pickle.load(f))
    f.close()
    f = open('/home/asr4/sea/dcase/workarea/DSS/traindata_9_t_3.pkl','rb')
    f3 = abs(pickle.load(f))
    f.close()
    f = open('/home/asr4/sea/dcase/workarea/DSS/traindata_9_t_4.pkl','rb')
    f4 = abs(pickle.load(f))
    f.close()
    f = open('/home/asr4/sea/dcase/workarea/DSS/traindata_9_t_5.pkl','rb')
    f5 = abs(pickle.load(f))
    f.close()
    f = open('/home/asr4/sea/dcase/workarea/DSS/traindata_9_t_6.pkl','rb')
    f6 = abs(pickle.load(f))
    f.close()
    print('data load')
    # all_features = abs(np.concatenate((f1,f2,f3,f4,f5,f6),axis=0))
    all_features = np.concatenate((f1,f2,f3,f4,f5,f6),axis=0)
    # all_label = np.array(np.load('label.npy'))
    all_features = torch.from_numpy(all_features)
    # all_label = torch.from_numpy(all_label)
    # all_features = all_features.unsqueeze(1)
    all_features = all_features.float()
    all_features = all_features.permute(0,3,1,2)
    print(all_features.shape)
    # all_label = all_label.long()
    train_f = all_features
    f = open('/home/asr4/sea/dcase/workarea/DSS/testdata_9_t_1.pkl','rb')
    t1 = abs(pickle.load(f))
    f.close()
    f = open('/home/asr4/sea/dcase/workarea/DSS/testdata_9_t_2.pkl','rb')
    t2 = abs(pickle.load(f))
    f.close()
    test_f = np.concatenate((t1,t2),axis=0)
    test_f = torch.from_numpy(test_f).float()
    test_f = test_f.permute(0,3,1,2)
    print(test_f.shape)
    f = open('/home/asr4/sea/dcase/workarea/DSS/trainlabel_9_t.pkl','rb')
    train_onehot = pickle.load(f)
    f.close()
    train_onehot = train_onehot.tolist()
    train_l = []
    for tr in range(len(train_onehot)):
        for j in train_onehot[tr]:
            if j == 1:
                train_l.append(train_onehot[tr].index(j))
    train_l = torch.from_numpy(np.array(train_l))
    f = open('/home/asr4/sea/dcase/workarea/DSS/testlabel_9_t.pkl','rb')
    test_onehot = pickle.load(f)
    f.close()
    test_onehot = test_onehot.tolist()
    test_l = []
    for tr in range(len(test_onehot)):
        for j in test_onehot[tr]:
            if j == 1:
                test_l.append(test_onehot[tr].index(j))
    test_l = torch.from_numpy(np.array(test_l))
    a = [0 for _ in range(10)]
    for i in test_l.numpy().tolist():
        a[i]+=1
    print(a)
    # print(test_l.numpy().tolist()[-400:])
    # print(all_features.shape, all_label.shape)
    return train_f,train_l,test_f,test_l

def get_data_7200():
    f = open('/home/asr4/sea/dcase/workarea/DSS/evaluation_data_9_t_1.pkl','rb')
    f1 = pickle.load(f)
    f.close()
    f = open('/home/asr4/sea/dcase/workarea/DSS/evaluation_data_9_t_2.pkl','rb')
    f2 = pickle.load(f)
    f.close()
    f = open('/home/asr4/sea/dcase/workarea/DSS/evaluation_data_9_t_3.pkl','rb')
    f3 = pickle.load(f)
    f.close()
    f = open('/home/asr4/sea/dcase/workarea/DSS/evaluation_data_9_t_4.pkl','rb')
    f4 = pickle.load(f)
    f.close()
    features = np.concatenate((f1,f2,f3,f4),axis=0)
    features = torch.from_numpy(features).float()
    features = features.permute(0,3,1,2)
    print(features.shape)
    return features

if __name__ == '__main__':
    LR = 0.001
    EPOCH = 80
    BATCHSIZE = 128
    train_data, train_label, test_data, test_label = get_data()
    # print(type(train_data),type(label))
    dataset = Data.TensorDataset(train_data, train_label)
    train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)
    # test_data = []
    # test_loader = data.DataLoader(test_data, batch_size=100, shuffle=True)
    model = CNN()
    model.train()
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    # loss_old = 100
    #try:
    #    os.remove('CNN3.pt')
    #except:
    #    pass
    for epoch in range(EPOCH):
        for i,(x,y) in enumerate(train_loader):
            #if os.path.exists('CNN3.pt'):
                # model = CNN()
                # model.load_state_dict(torch.load('CNN3.pt'))
                # model.eval()
                #out = model(x)
            #else:
            out = model(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if loss<loss_old:
            #    torch.save(model.state_dict(),'CNN3.pt')
            #    loss_old = loss
            if i%20 == 0:
                print('step:',i,'|loss:',loss.item())
                accuracy = torch.max(out, dim=1)[1].numpy() == y.numpy()
                accuracy = accuracy.mean()
                print(accuracy)
    torch.save(model, 'DSSCNNGRU_lucky.pkl')
    model.eval()
    state = model.state_dict()
    torch.save(state,'DSSCNNGRU2.pt')
    # model.eval()
    eva_data = get_data_7200()
    eva_dataset = Data.TensorDataset(eva_data,)
    eva_loader = Data.DataLoader(dataset=eva_dataset, batch_size=1)
    eva_out_softmax = []
    for i,(xe,) in enumerate(eva_loader):
        out_e = model(xe)
        out_e = F.softmax(out_e,dim=1)
        eva_out_softmax.append(out_e)
    eva_out_softmax = np.array(eva_out_softmax)
    count = 0
    test_dataset = Data.TensorDataset(test_data, test_label)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=1)
    # out_softmax = []
    for i,(xt,yt) in enumerate(test_loader):
        out = model(xt)
        flag = torch.max(out, dim=1)[1].numpy() == yt.numpy()
        if flag == True:
            count+=1
        # out_s = F.softmax(out,dim=1)
        # out_s = out_s.detach().numpy().tolist()[0]
        # out_softmax.append(out_s)
    print('+++++',count/2880)
    # out_softmax = np.array(out_softmax)
    # print(out_softmax.shape)
    # np.save('train_2880_softmax.npy',out_softmax)
    # eva_data = get_data_7200()
    # eva_dataset = Data.TensorDataset(eva_data,)
    # eva_loader = Data.DataLoader(dataset=eva_dataset, batch_size=1)
    # eva_out_softmax = []
    # for i in range(7200):
    #for i,(xe,) in enumerate(eva_loader):
    #    out_e = model(eva_data[i])
    #    out_e = F.softmax(out_e,dim=1)
    #    out_e = out_e.detach().numpy().tolist()[0]
    #    eva_out_softmax.append(out_e)
    eva_out_softmax = np.array(eva_out_softmax)
    print(eva_out_softmax.shape)
    np.save('train_7200_softmax.npy', eva_out_softmax)
            
