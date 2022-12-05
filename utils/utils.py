import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np


class argparse():
    batch_size = 32
    epochs = 200
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_seed(seed=42):
    import random
    # seed = 42  # 生命、宇宙和一切终极问题的答案
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class myLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.maeloss = myMAELoss()
        self.alpha = alpha

    def forward(self, input, target):
        #print(input.shape, target.shape)
        theta = torch.abs(torch.abs(input - target)[:, 0] - torch.abs(input - target)[:, 1])
        # print(theta)
        return self.maeloss(input, target) + torch.mean(theta) * self.alpha


class myMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = nn.MSELoss(reduction='mean')
        self.loss2 = nn.MSELoss(reduction='mean')

    def forward(self, input, target):

        return torch.sum(self.loss1(input[:, 0], target[:, 0]) + self.loss2(input[:, 1], target[:, 1]))


class myMAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = nn.L1Loss(reduction='mean')
        self.loss2 = nn.L1Loss(reduction='mean')

    def forward(self, input, target):

        # return (self.loss1(input[:, 0], target[:, 0]) + self.loss2(input[:, 1], target[:, 1])) / 2
        return (self.loss1(input[:, 0], target[:, 0]) + self.loss2(input[:, 1], target[:, 1])) 


class MASELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSELoss = myMSELoss()
        self.MAELoss = myMAELoss()

    def forward(self, input, target):

        return self.MSELoss(input, target) + self.MAELoss(input, target)


def load_weights(model, f):
    print('loading weights')
    weight = torch.load(f)
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in weight.items(
    ) if k in model_state_dict.keys()}
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print('finish')


def draw_loss(train_epochs_loss, valid_epochs_loss, fig_name='output/graph/valid/loss.png'):
    plt.figure(figsize=(8, 8))

    plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
    plt.title("each_epochs_loss")
    #plt.ylim(ymin = 0, ymax=2000)
    plt.legend()

    plt.savefig(fig_name)
    plt.show()

# 评估图


def draw_valid(MSE_val, RMSE_val, MAE_val, fig_name='output/graph/valid/valid.png'):
    plt.figure(figsize=(16, 16))
    # Horizontal displacement 水平  vertical 垂直
    MSE_val = np.array(MSE_val)
    RMSE_val = np.array(RMSE_val)
    MAE_val = np.array(MAE_val)

    plt.subplot(221)
    plt.ylabel('MSE')
    plt.xlabel('Valid data no.')
    plt.plot(MSE_val[:, 0], '-o', label='Horizontal displacement')
    plt.plot(MSE_val[:, 1], '-o', label='Vertical displacement')
    plt.title("MSE")
    plt.legend()

    plt.subplot(222)
    plt.ylabel('RMSE')
    plt.xlabel('Valid data no.')
    plt.plot(RMSE_val[:, 0], '-o', label='Horizontal displacement')
    plt.plot(RMSE_val[:, 1], '-o', label='Vertical displacement')
    plt.title("RMSE")
    plt.legend()

    plt.subplot(223)
    plt.ylabel('MAE')
    plt.xlabel('Valid data no.')
    plt.plot(MAE_val[:, 0], '-o', label='Horizontal displacement')
    plt.plot(MAE_val[:, 1], '-o', label='Vertical displacement')
    plt.title("MAE")
    plt.legend()

    plt.subplot(224)
    data = np.array([np.average(MAE_val), np.average(MSE_val), np.sqrt(np.average(np.square(MAE_val)))])
    labels = ['MAE', 'MSE', 'RMSE']
    plt.bar(labels, data, 0.5)

    plt.title("Average")

    plt.savefig(fig_name)
    plt.show()

# 学习率图


def draw_lr(lr, fig_name='output/graph/train/learning_rate.png'):
    plt.figure(figsize=(8, 8))
    lr = np.array(lr)

    plt.ylabel('Learnisng rate')
    plt.xlabel('Epoch')
    plt.plot(lr, '-o')

    plt.savefig(fig_name)
    plt.show()


# 测试误差图
def draw_error(gt, pred, fig_name='output/graph/train/error.png'):
    plt.figure(figsize=(16, 8))
    gt = np.array(gt)
    pred = np.array(pred)
    errors_h = gt[:, 0] - pred[:, 0]
    errors_v = gt[:, 1] - pred[:, 1]
    plt.subplot(221)
    plt.ylabel('displacement [mm]')
    plt.xlabel('Valid data no.')
    plt.plot(gt[:, 0], '-o', label='Ground truth')
    plt.plot(pred[:, 0], '-o', label='Predition')
    plt.title("Horizontal displacement")
    plt.legend()

    plt.subplot(222)
    plt.ylabel('displacement [mm]')
    plt.xlabel('Valid data no.')
    plt.plot(gt[:, 1], '-o', label='Ground truth')
    plt.plot(pred[:, 1], '-o', label='Predition')
    plt.title("Vertical displacement")

    plt.legend()
    plt.subplot(223)
    plt.ylabel('error [mm]')
    plt.xlabel('Valid data no.')
    plt.plot(errors_h, '-o', label='Ground truth')
    plt.axhline(y=1, color='red')
    plt.axhline(y=-1, color='red')
    plt.title("Vertical displacement")

    plt.legend()
    plt.subplot(224)
    plt.ylabel('error [mm]')
    plt.xlabel('Valid data no.')
    plt.plot(errors_v, '-o', label='Ground truth')
    plt.axhline(y=1, color='red')
    plt.axhline(y=-1, color='red')
    plt.title("Vertical displacement")

    plt.legend()
    plt.savefig(fig_name)
    plt.show()


def draw_pic(pics, fig_name='data_graph.png'):
    plt.figure(figsize=(12, 4))
    title = 'CNN input (Linear input:' + str(pics[-1][0]) + ' ' + str(pics[-1][1]) + ' ' + str(pics[-1][2]) + ')'
    plt.suptitle(title)

    plt.subplot(133)
    im1 = plt.imshow(pics[0])
    plt.colorbar(im1, fraction=0.05, pad=0.05)
    plt.title("Linear layer output")

    plt.subplot(131)
    im2 = plt.imshow(pics[1])
    plt.colorbar(im2, fraction=0.05, pad=0.05)
    plt.title("data graph 1")

    plt.subplot(132)
    im3 = plt.imshow(pics[2])
    plt.colorbar(im3, fraction=0.05, pad=0.05)
    plt.title("data graph 2")

    plt.savefig(fig_name)
    plt.show()


def calculate_evaluation_metrics(gt, pred):
    mse = (gt - pred)**2
    rmse = mse**0.5
    mae = np.abs(gt - pred)
    return mse, rmse, mae
