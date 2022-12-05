import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.dataloader import myDataSet
from utils.utils import *
from models.model import *
from models.conv_1 import *
from models.otherModel import *


# Initialize the random seed
init_seed()

# Hyperparameters
args = argparse()
args.batch_size = 1
args.epochs = 1

# Get the dataset
valData = myDataSet(train = False, vaild = False, extra = True)
valLoader = DataLoader(valData, batch_size = args.batch_size, shuffle = False)


dir_name = 'final_linear'
# model_name = r'../models/' + dir_name + '/' + dir_name + '_best.pth'
# output_dir = r'../models/' + dir_name + '/graph/test/'
model_name = r'output/' + dir_name + '/' + dir_name + '_best.pth'
output_dir = 'output/' + dir_name + '/graph/test/'
# model_name = r'output/resnet_34v3_150.pth'

# Initialize the model
model = Linear().to(args.device)
load_weights(model, model_name)
model.cuda()
# ----------------------------------------------------------------


# Initialize the variable
RMSE_val = []
MSE_val = []
MAE_val = []
gt = []
pred = []
errors = []
# ---------------------------------------------------------------------


# 

try:
    model.eval()
    with torch.no_grad():
        print('start valid')

        for epoch in range(args.epochs):
            # =====================valid============================
            print('-------------------valid-----------------')

            valid_epoch_loss = []
            for idx, (x1, x2, y) in enumerate(valLoader):
                if (idx) % 50 == 0:

                    x1, x2 = Variable(x1).cuda(), Variable(
                        x2).cuda()
                    print(x1.shape, x2.shape)
                    yHat = model((x1, x2)).cpu()
                    yHat = torch.clamp(yHat, min = 0)
                    gt.append(y[0].numpy())
                    pred.append(yHat[0].numpy())

                    inp = x2[0].cpu().numpy()

                    # draw_pic([model.pic[0], x1[0, 0, :].cpu().numpy(), 
                    #           x1[0, 1, :].cpu().numpy(), inp],
                    #          fig_name=output_dir + 'data_graph.png')
                    # draw_pic([model.pic[0][:, :, 0], model.pic[0][:, :, 1], 
                    #          model.pic[0][:, :, 2], inp], fig_name = output_dir + 'data_graph.png')

                    mse, rmse, mae = calculate_evaluation_metrics(y[0].numpy(), yHat[0].numpy())

                    RMSE_val.append(rmse)
                    MSE_val.append(mse)
                    MAE_val.append(mae)
                    error = yHat[0] - y[0]
                    errors.append(error)
                    print('------------------batch:{}---------------'.format(idx + 1))
                    print(' | real | pred | error | mse | rmse | mae |')
                    print("x|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|".format(
                        y[0][0], yHat[0][0], error[0], mse[0], rmse[0], mae[0]))
                    print("y|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|\n".format(
                        y[0][1], yHat[0][1], error[1], mse[1], rmse[1], mae[1]))

    # Plot the loss of the training process
    draw_valid(MSE_val, RMSE_val, MAE_val, fig_name = output_dir + 'valid.png')
    draw_error(gt, pred, fig_name = output_dir + 'error.png')


except KeyboardInterrupt:
    draw_valid(MSE_val, RMSE_val, MAE_val, fig_name = output_dir + 'valid.png')
    draw_error(gt, pred, fig_name = output_dir + 'error.png')
    print('stop')
