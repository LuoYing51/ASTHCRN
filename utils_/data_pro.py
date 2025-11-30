import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset

def dataprox(args, data):
    tim_len = data.shape[0]
    pre_len = args.output_T_dim
    seq_len = args.T_dim
    train_rate = args.train_rate
    trainX, trainY, testX, testY = preprocess_data(data, time_len=tim_len, train_rate=train_rate, seq_len=seq_len,
                                                   pre_len=pre_len)
    return trainX, trainY, testX, testY


def preprocess_data(data, time_len, train_rate, seq_len, pre_len):
    train_size = int(time_len * train_rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len, :, 5])

    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len, :, 5])

    trainX = np.array(trainX)
    trainY = np.array(trainY).transpose(0, 2, 1)
    testX = np.array(testX)
    testY = np.array(testY).transpose(0, 2, 1)

    return trainX, trainY, testX, testY


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = torch.tensor(X, dtype=torch.float32, device='cuda'), torch.tensor(Y, dtype=torch.float32, device='cuda')

    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
