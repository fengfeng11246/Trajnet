import torch
import numpy as np
from torch.autograd import Variable
import argparse
import os
import time
import pickle
import subprocess
from losses import *
from Data_utils import *
from model import *

parser = argparse.ArgumentParser()
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--input_size', type=int, default=2)  # 输入坐标，x,y
parser.add_argument('--output_size', type=int, default=2)
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
# Size of each batch parameter
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
# Length of sequence to be considered parameter
parser.add_argument('--seq_length', type=int, default=20,
                    help='RNN sequence length')  # 输入长度
parser.add_argument('--pred_length', type=int, default=12,
                    help='prediction length')  # 输出长度
# Number of epochs parameter
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs')
# Frequency at which the model should be saved parameter
parser.add_argument('--save_every', type=int, default=400,
                    help='save frequency')
# TODO: (resolve) Clipping gradients for now. No idea whether we should
# Gradient value at which it should be clipped
parser.add_argument('--grad_clip', type=float, default=10.,
                    help='clip gradients at this value')
# Learning rate parameter
parser.add_argument('--learning_rate', type=float, default=0.003,
                    help='learning rate')
# Decay rate for the learning rate parameter
parser.add_argument('--decay_rate', type=float, default=0.95,
                    help='decay rate for rmsprop')
# Dropout not implemented.
# Dropout probability parameter
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout probability')
# Dimension of the embeddings parameter
parser.add_argument('--embedding_size', type=int, default=64,
                    help='Embedding dimension for the spatial coordinates')  # 怎么确定的？64，embedding层怎么用？
# Size of neighborhood to be considered parameter
parser.add_argument('--neighborhood_size', type=int, default=32,
                    help='Neighborhood size to be considered for social grid')  # pool层的大小？
# Size of the social grid parameter
parser.add_argument('--grid_size', type=int, default=4,
                    help='Grid size of the social grid')  # 什么东西？人与人之间距离？
# Maximum number of pedestrians to be considered
parser.add_argument('--maxNumPeds', type=int, default=27,
                    help='Maximum Number of Pedestrians')
# Lambda regularization parameter (L2)
parser.add_argument('--lambda_param', type=float, default=0.0005,
                    help='L2 regularization parameter')
# Cuda parameter
parser.add_argument('--use_cuda', action="store_true", default=False,
                    help='Use GPU or not')
# GRU parameter
parser.add_argument('--gru', action="store_true", default=False,
                    help='True : GRU cell, False: LSTM cell')
# number of validation will be used
parser.add_argument('--num_validation', type=int, default=2,
                    help='Total number of validation dataset for validate accuracy')
# frequency of validation
parser.add_argument('--freq_validation', type=int, default=10,
                    help='Frequency number(epoch) of validation using validation data')
# frequency of optimazer learning decay
parser.add_argument('--freq_optimizer', type=int, default=8,
                    help='Frequency number(epoch) of learning decay for optimizer')
# store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
parser.add_argument('--grid', action="store_true", default=True,
                    help='Whether store grids and use further epoch')
parser.add_argument('--gpu_num', default="3,4,5", type=str)
# multi-model run
parser.add_argument('--model_type', default=1, type=int)
#the method of preprocessing
parser.add_argument('--model_processing', default=0, type=int)
args = parser.parse_args()
#prepare
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
model_NO = []
methods = []
#get the list of  model type
while (args.model_type>0):
    model_NO.append(args.model_type %10)
    args.model_type = int(args.model_type/10)
model_NO = list(reversed(model_NO))
# get the list of preprocessing method
while (args.model_processing>0):
    methods.append(args.model_processing %10)
    args.model_processing = int(args.model_processing/10)
methods = list(reversed(methods))

args.freq_validation = np.clip(args.freq_validation, 0, args.num_epochs)
validation_epoch_list = list(range(args.freq_validation, args.num_epochs+1, args.freq_validation))
validation_epoch_list[-1]-=1
#load data
data = Data_utils(args.batch_size, args.seq_length, args.num_validation, infer = True,)
data_path = data.Datapath()
# data preproces
#method:1--rotate,2--reverse, 3--offset,4-- normalization, 5--standardzation 6
data_pro = data.Dataprocess(data_path,methods)

x_train = []
x_pred = []
for line_n,line in enumerate(data_pro):
    if line_n%20 in range(0, 8):  # 0~7 # when in observation sequence
        x_train.append(line)
    else: # when in prediction sequence
        x_pred.append(line)
num_batches = int(len(data_pro)/(args.batch_size*args.seq_length))
data_count = len(x_train)/8
x_train = np.array(x_train)
x_label = np.array(x_pred)

for num in model_NO:
    if num == 1:
        print("********** SOCIAL LSTM ******************")
        model_name = "LSTM"
        method_name = "dual_LSTM"
        args.output_size = 2
        save_tar_name = method_name + "_lstm_model_"
        if args.gru:
            model_name = "GRU"
            save_tar_name = method_name + "_gru_model_"
        # Log directory
        log_directory = os.path.join( 'log/')
        plot_directory = os.path.join('plot/', method_name, model_name)
        plot_train_file_directory = 'validation'

        # Logging files
        dir_logname = os.path.join(log_directory, method_name, model_name)

        if not os.path.isdir(dir_logname):
            os.makedirs(dir_logname)
        log_file_curve = open(os.path.join(log_directory, method_name, model_name, 'log_curve.txt'), 'w+')
        log_file = open(os.path.join(log_directory, method_name, model_name, 'val.txt'), 'w+')

        # model directory
        save_directory = os.path.join( 'model/')

        # Save the arguments int the config file
        dir_savename = os.path.join(save_directory, method_name, model_name)
        if not os.path.isdir(dir_savename):
            os.makedirs(dir_savename)
        with open(os.path.join(save_directory, method_name, model_name, 'config.pkl'), 'wb') as f:
            pickle.dump(args, f)


        # Path to store the checkpoint file
        def checkpoint_path(x):
            return os.path.join(save_directory, method_name, model_name, save_tar_name + str(x) + '.tar')
        # model creation
        net = Dual_Lstm()



        if args.use_cuda:
            net = net.cuda()
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
        optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
        # optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)
        loss_function = nn.MSELoss()

        learning_rate = args.learning_rate
        # Training
        for epoch in range(args.num_epochs):
            print('****************Training epoch beginning******************')
            loss_epoch = 0
            batch_train = np.zeros((args.seq_length-args.pred_length,4))
            batch_pred = np.zeros((args.pred_length,4))


            for batch in range(num_batches):
                start = time.time()
                loss_batch = 0
                batchcount_obj = batch*args.batch_size*8
                batchcount_pred = batch*args.batch_size*12
                # For each batch
                for sequence in range(args.batch_size):
                    # Get the data corresponding to the current sequence
                    count_obj = batchcount_obj + sequence*8
                    count_pred = batchcount_pred + sequence*12
                    a= len(x_train)
                    b= len(x_label)
                    count_obj_right = min(count_obj+8,len(x_train)-1)
                    count_pred_right = min(count_pred + 12, len(x_label)-1)

                    x_seq = x_train[count_obj:count_obj_right,2:]
                    x_pred = x_label[count_pred:count_pred_right,2:]


                    x_seq = Variable(torch.from_numpy(x_seq).float().view(-1,len(x_seq),2))
                    x_pred = Variable(torch.from_numpy(x_pred).float())




                    if args.use_cuda:
                        x_seq = x_seq.cuda()

                    # Zero out gradients
                    net.zero_grad()
                    optimizer.zero_grad()
                    # Forward prop
                    outputs = net(x_seq)

                    # Compute loss
                    loss = loss_function(outputs,x_pred)

                    loss_batch += loss
                    # loss_batch=loss

                    # Compute gradients
                    loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                    # Update parameters
                    optimizer.step()

                end = time.time()
                loss_batch = loss_batch / args.batch_size
                loss_epoch += loss_batch


                print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(
                    epoch * num_batches + batch,
                    args.num_epochs * num_batches,
                    epoch,
                    loss_batch, end - start))

            loss_epoch /= num_batches
            # Log loss values
            log_file_curve.write("Training epoch: " + str(epoch) + " loss: " + str(loss_epoch) + '\n')

            # Save the model after each epoch
            print('Saving model')
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path(epoch))

    # Close logging files
    log_file.close()
    log_file_curve.close()



#data for training
#data_input = data.DataInput()
