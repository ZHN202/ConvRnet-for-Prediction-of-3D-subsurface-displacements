import math
import os
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.dataloader import myDataSet
from utils.utils import *
from models.model import *
from models.conv_1 import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.otherModel import *

# Initialize the random seed
init_seed(seed=0)

# Hyperparameters
args = argparse()
args.batch_size = 51
args.epochs = 500
args.learning_rate = 0.001

# Get the dataset
trainData = myDataSet()
valData = myDataSet(train=False, vaild=False, extra=True)
trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=False)
valLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=True)


save_model_name = 'final_conv1d_'


if not os.path.exists('output/' + save_model_name[:-1]):
    os.mkdir('output/' + save_model_name[:-1])
    os.mkdir('output/' + save_model_name[:-1] + '/graph')
    os.mkdir('output/' + save_model_name[:-1] + '/graph/train')
    os.mkdir('output/' + save_model_name[:-1] + '/graph/test')
    os.mkdir('output/' + save_model_name[:-1] + '/log')

output_dir = 'output/' + save_model_name[:-1] + '/'
# Initialize the model
model = Conv1d().to(args.device)
if os.path.exists(output_dir + save_model_name + 'last.pth'):
    load_weights(model, output_dir + save_model_name + 'last.pth')
model.cuda()
# ----------------------------------------------------------------


# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
# optimizer = torch.optim.SGD(
#    model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=1, eta_min=0)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)
criterion = myLoss(alpha=1).to(args.device)
# -----------------------------------------------------------------


# Initialize the variable
train_epochs_loss = []
valid_epochs_loss = []
lr = []
valid_best = 50
# ---------------------------------------------------------------------

f = open('output/' + save_model_name[:-1] + '/log' + save_model_name + 'log.txt', 'w+')

# train

try:
    print('start train')
    for epoch in range(args.epochs):

        train_epoch_loss = []
        valid_fold_loss = []

        # 17folds
        for i in range(17):

            # Training for every fold
            print('----------------train-------------------' + save_model_name)
            model.train()
            train_batch_loss = []
            for idx, (x1, x2, y) in enumerate(trainLoader):
                if i * 51 <= idx < (i + 1) * 51:
                    continue

                x1, x2, y = Variable(x1).cuda(), Variable(
                    x2).cuda(), Variable(y).cuda()
                yHat = model((x1, x2))
                # yHat=yHat.view(args.batch_size,2)
                loss = criterion(yHat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_batch_loss.append(loss.item())

            train_epoch_loss.append(np.average(train_batch_loss))

            # Test of every fold
            print('-------------------valid-----------------')
            model.eval()
            valid_batch_loss = []
            for idx, (x1, x2, y) in enumerate(trainLoader):
                if i * 51 <= idx < (i + 1) * 51:
                    x1[:, 2, :, :] = x1[:, 2, :, :] 
                    x2[:, 2] = x2[:, 2] 
                    x1, x2, y = Variable(x1).cuda(), Variable(
                        x2).cuda(), Variable(y).cuda()
                    yHat = model((x1, x2))
                    # yHat=yHat.view(args.batch_size,2)
                    loss = criterion(yHat, y)

                    valid_batch_loss.append(loss.item())
            valid_fold_loss.append(np.average(valid_batch_loss))

        # After 17 folds, train on all datasets
        train_loss_after_fold = []
        for idx, (x1, x2, y) in enumerate(trainLoader):

            x1, x2, y = Variable(x1).cuda(), Variable(
                x2).cuda(), Variable(y).cuda()
            yHat = model((x1, x2))
            loss = criterion(yHat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_after_fold.append(loss.item())

        train_epoch_loss.append(np.average(train_loss_after_fold))

        print("epoch={}/{}, loss={:.2f}\n".format(
            epoch + 1, args.epochs, np.average(train_loss_after_fold)))
        f.write("epoch={}/{}, loss={:.2f}\n".format(
                epoch + 1, args.epochs, np.average(train_loss_after_fold)))

        # Adjust the learning rate
        lr.append(scheduler.get_last_lr())
        scheduler.step()

        print("epoch={}/{} of train, average train loss={:.2f}, average valid loss={:.2f}".format(
            epoch + 1, args.epochs, np.average(train_epoch_loss), np.average(valid_fold_loss)))
        f.write("epoch={}/{} of train, average train loss={:.2f}, average valid loss={:.2f}\n".format(
            epoch + 1, args.epochs, np.average(train_epoch_loss), np.average(valid_fold_loss)))

        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================test============================
        print('\n----------------test-best:' + str(valid_best) + '-------------------')
        model.eval()
        valid_epoch_loss = []
        for idx, (x1, x2, y) in enumerate(valLoader):

            x1, x2, y = Variable(x1).cuda(), Variable(
                x2).cuda(), Variable(y).cuda()
            yHat = model((x1, x2))
            # yHat=yHat.view(args.batch_size,2)
            loss = criterion(yHat, y)
            valid_epoch_loss.append(loss.item())

        print("epoch={}/{} of test, average loss={:.2f}".format(
            epoch + 1, args.epochs, np.average(valid_epoch_loss)))
        f.write("epoch={}/{} of test, average loss={:.2f}".format(
            epoch + 1, args.epochs, np.average(valid_epoch_loss)))
        valid_epochs_loss.append(np.average(valid_epoch_loss))

        # Save the optimal weight
        if np.average(valid_epoch_loss) < valid_best:   
            valid_best = np.average(valid_epoch_loss)
            torch.save(model.state_dict(), output_dir +
                       save_model_name + 'best.pth')

    # Draw the loss graph of the training process
    print('finish')
    print('saving the last model')
    torch.save(model.state_dict(), output_dir + save_model_name + 'last.pth')
    draw_loss(train_epochs_loss, 
              valid_epochs_loss, fig_name=output_dir + 'graph/train/loss.png')
    draw_lr(lr, fig_name=output_dir + 'graph/train/learning_rate.png')
    f.close()


except KeyboardInterrupt:
    print('stop')
    print('saving the last model')
    f.close()
    torch.save(model.state_dict(), output_dir + save_model_name + 'last.pth')
    draw_loss(train_epochs_loss, 
              valid_epochs_loss, fig_name=output_dir + 'graph/train/loss.png')
    draw_lr(lr, fig_name=output_dir + 'graph/train/learning_rate.png')
