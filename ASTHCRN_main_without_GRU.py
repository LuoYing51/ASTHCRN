import argparse
import math

import pandas as pd
import numpy as np
import torch
from torch import nn

from ASTHCRN.utils_.data_pro import data_loader
from ASTHCRN.ablation.ASTHCRN_model_without_GRU import main
from ASTHCRN.utils_.All_Metrics import All_Metrics

import torch.optim as optim
from datetime import datetime
import time
from ASTHCRN.utils_.get_logger import get_logger
import os
import copy
from ASTHCRN.utils_.inverse_normalize import inverse_normalize
from ASTHCRN.utils_.propera_data import preprocess_data_day
from ASTHCRN.utils_.init_seed import init_seed
from ASTHCRN.utils_.save_result import save_result_args
from sklearn.preprocessing import MinMaxScaler

MODEL = 'ASTHCRN_without_GRU'
DEBUG = 'False'
dataset_name="dataset"

device = torch.device('cuda:0')
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--train_rate', default=0.6, type=float, help='rate of train set.')
args.add_argument('--val_rate', default=0.2, type=float, help='rate of val set.')
args.add_argument('--test_rate', default=0.2, type=float, help='rate of test set.')
args.add_argument('--T_dim', default=24, type=int, help='time length of inputs.')
args.add_argument('--output_T_dim', default=12, type=int)

args.add_argument('--num_nodes', default=16, type=int)#The value in the virtual data is 16
args.add_argument('--batch_size', default=16, type=int)
args.add_argument('--in_channels', default=1, type=int)
args.add_argument('--output_dim', default=1, type=int)
args.add_argument('--learning_rate', default=0.005, type=float)

args.add_argument('--embed_size', default=64, type=int)
args.add_argument('--num_layers', default=2, type=int)
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--epochs', default=1, type=int)
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=10, type=int)

args.add_argument('--seed', default=42, type=int)
args.add_argument('--dropout', default=0.1, type=float)
args.add_argument('--d_inner', type=int, default=64)

args.add_argument('--HGCNADP_topk', type=int)
args.add_argument('--hyperedge_rate', default=0.3, type=float)
args.add_argument('--HGCNADP_embed_dims', default=40, type=int)

args = args.parse_args()
init_seed(args.seed)

torch.cuda.synchronize()
args.HGCNADP_topk = math.ceil(args.hyperedge_rate * args.num_nodes)

# load dataset
dataframe = pd.read_excel(
    r'../ASTHCRN/data/data.xlsx',)
dataset_aqi = dataframe.values
dataset = dataset_aqi[:, -2]
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (args.num_nodes, -1, 1))
data = dataset.transpose(1, 0, 2)

original_data = data.copy()
data_flat = data.reshape(-1, 1)
train_time_steps = int(data.shape[0] * args.train_rate)
train_indices = train_time_steps * args.num_nodes
scaler = MinMaxScaler()
scaler.fit(data_flat[:train_indices])
normalized_data_flat = scaler.transform(data_flat)
data = normalized_data_flat.reshape(data.shape)

if args.val_rate == 0.0:
    train_size = int(data.shape[0] * args.train_rate)
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_X, train_Y = preprocess_data_day(train_data, args.T_dim, args.output_T_dim)
    test_X, test_Y = preprocess_data_day(test_data, args.T_dim, args.output_T_dim)
    train_loader = data_loader(train_X, train_Y, args.batch_size, shuffle=True,
                               drop_last=False)
    test_loader = data_loader(test_X, test_Y, args.batch_size, shuffle=False,
                              drop_last=False)
    val_loader = None

else:
    train_size = int(data.shape[0] * args.train_rate)
    val_size = int(data.shape[0] * args.val_rate)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    train_X, train_Y = preprocess_data_day(train_data, args.T_dim, args.output_T_dim)
    test_X, test_Y = preprocess_data_day(test_data, args.T_dim, args.output_T_dim)
    val_X, val_Y = preprocess_data_day(val_data, args.T_dim, args.output_T_dim)

    train_loader = data_loader(train_X, train_Y, args.batch_size, shuffle=True, drop_last=False)
    test_loader = data_loader(test_X, test_Y, args.batch_size, shuffle=False, drop_last=False)
    val_loader = data_loader(val_X, val_Y, args.batch_size, shuffle=False, drop_last=False)

### Construct Network
net = main(
    args.in_channels,
    args.output_dim,
    args.embed_size,
    args.d_inner,
    args.num_layers,
    args.T_dim,
    args.output_T_dim,
    args.num_nodes,
    device,
    HGCNADP_topk=args.HGCNADP_topk,
    hyperedge_rate=args.hyperedge_rate,
    HGCNADP_embed_dims=args.HGCNADP_embed_dims,
    dropout=args.dropout)

for p in net.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
net = net.to(device)

current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
name = 'in_{}h_out_{}h'.format(args.T_dim, args.output_T_dim)
log_dir = os.path.join(current_dir, 'ASTHCRN_'+dataset_name, name)

print(log_dir)
args.log_dir = log_dir
if os.path.isdir(args.log_dir) == False and not args.debug:
    os.makedirs(args.log_dir, exist_ok=True)
logger = get_logger(args.log_dir, name=MODEL, debug=args.debug)
logger.info('Experiment log path in: {}'.format(args.log_dir))

criterion = torch.nn.L1Loss().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
start_time = time.time()

best_loss = np.inf
not_improved_count = 0
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()


    total_loss = 0

    net.train()

    for batch_index, batch_data in enumerate(train_loader):
        train_per_epoch = len(train_loader)
        train_X, train_Y = batch_data
        train_X = train_X.to(device)
        train_Y = train_Y.to(device)

        optimizer.zero_grad()
        train_output = net(train_X)  # [B,T,N,dim]
        loss = criterion(train_output, train_Y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_index % args.log_step == 0:
            logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                epoch, batch_index, train_per_epoch, loss.item()))
        # train_outputs = torch.cat((train_outputs,train_output),0)
    train_epoch_loss = total_loss / train_per_epoch
    logger.info(
        '**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
    # val
    if val_loader == None:
        val_dataloader = test_loader
    else:
        val_dataloader = val_loader

    net.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            val_X1, val_Y = batch_data
            val_X1 = val_X1.to(device)

            val_Y = val_Y.to(device)
            output = net(val_X1)
            loss = criterion(output, val_Y)
            if not torch.isnan(loss):
                total_val_loss += loss.item()
    val_epoch_loss = total_val_loss / len(val_dataloader)
    logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_epoch_loss))

    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        not_improved_count = 0
        best_state = True
    else:
        not_improved_count += 1
        best_state = False

    if args.early_stop:
        if not_improved_count == args.early_stop_patience:
            print("Validation performance didn\'t improve for {} epochs. "
                  "Training stops.".format(args.early_stop_patience))
            break
    if best_state == True:
        print('*********************************Current best model saved!')
        best_model = copy.deepcopy(net.state_dict())

training_time = time.time() - start_time
logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

torch.save(best_model, '{}/best_model.pth'.format(log_dir))
logger.info("Saving current best model to " + '{}/best_model.pth'.format(log_dir))

net.load_state_dict(best_model)

net.eval()
test_outputs = []
test_targets = []

with torch.no_grad():
    for batch_data in test_loader:
        test_X, test_Y = batch_data
        test_X = test_X.to(device)

        pred = net(test_X)
        test_outputs.append(pred.cpu().numpy())
        test_targets.append(test_Y.cpu().numpy())

test_outputs = np.concatenate(test_outputs, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

denorm_pred = torch.from_numpy(inverse_normalize(test_outputs, scaler)).float().to(device)
denorm_true = torch.from_numpy(inverse_normalize(test_targets, scaler)).float().to(device)
mae, rmse, mape, mse, r2 = All_Metrics(denorm_pred, denorm_true)

for i in range(denorm_true.shape[1]):

    maes, rmses, mapes, mses, r2_s = All_Metrics(denorm_pred[:, i, :, :], denorm_true[:, i, :, :])
    logger.info(
        "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, MSE:{:.4f}, R2:{:.4f}".format(i + 1, maes, rmses,
                                                                                                mapes, mses, r2_s))
logger.info('mae1:{},rmse1:{},mape1:{},mse:{},r2_1:{}'.format(mae, rmse, mape, mse, r2))
result_txt = 'MAE:{}   ,RMSE:{}   ,MAPE:{}   ,MSE:{},     R2:{}'.format(mae, rmse, mape, mse, r2)

#Save the results
saved_path = save_result_args(
    result_content=result_txt,
    args=args,
    file_path=os.path.join(args.log_dir, name + "_result.txt"))
