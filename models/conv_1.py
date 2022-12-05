import torch.nn as nn
import torch
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.CBAM import cbam_block


class resnet18_with_attention_final(nn.Module):
    def __init__(self):
        super(resnet18_with_attention_final, self).__init__()
        self.pic = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 867, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(867),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)
        # print(x1[0,2,0,0],x2[0,2])

        conv1_out = self.conv1(x2)
        # print(conv1_out.shape)
        conv1_out = conv1_out.view(-1, 1, 2601).view(-1, 1, 51, 51)

        x1 = torch.cat((x1, conv1_out), 1)

        # print(x1.shape)
        x1 = self.conv2d_1(x1)
        x1 = self.cbam1(x1)
        # print(self.pic.shape)
        # self.pic = x1.detach().cpu().permute(0, 2, 3, 1).numpy()
        x1 = self.residual_block_1(x1)
        x1 = self.cbam2(x1)
        x1 = self.residual_block_2(x1)
        x1 = self.cbam3(x1)
        x1 = self.residual_block_3(x1)
        x1 = self.cbam4(x1)
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)
        out = self.out(x1)
        return out


class resnet18_with_attention_final_linear(nn.Module):
    def __init__(self):
        super(resnet18_with_attention_final_linear, self).__init__()
        self.pic = None

        self.conv1 = nn.Sequential(
            nn.Linear(3, 2601),
            nn.BatchNorm1d(2601),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1]
        # print(x1[0,2,0,0],x2[0,2])

        conv1_out = self.conv1(x2)
        # print(conv1_out.shape)
        conv1_out = conv1_out.view(-1, 1, 2601).view(-1, 1, 51, 51)

        im = x1[0, 1, :].detach().cpu().numpy()

        fig = plt.figure(figsize=(8, 8))
        print(im.shape)
        im1 = plt.imshow(im)
        plt.colorbar(im1, fraction=0.05, pad=0.05)
        # plt.imshow(self.pic[0])
        # plt.imshow(im[0])
        fig.savefig('U_II.pdf', dpi=600)

        x1 = torch.cat((x1, conv1_out), 1)

        # print(x1.shape)
        x1 = self.conv2d_1(x1)
        x1 = self.cbam1(x1)
        # print(self.pic.shape)
        # self.pic = x1.detach().cpu().permute(0, 2, 3, 1).numpy()
        x1 = self.residual_block_1(x1)
        x1 = self.cbam2(x1)
        x1 = self.residual_block_2(x1)
        x1 = self.cbam3(x1)
        x1 = self.residual_block_3(x1)
        x1 = self.cbam4(x1)
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)
        out = self.out(x1)
        return out


class resnet18_with_attention_final_without_CBAM(nn.Module):
    def __init__(self):
        super(resnet18_with_attention_final_without_CBAM, self).__init__()
        self.pic = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 867, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(867),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)
        # print(x1[0,2,0,0],x2[0,2])

        conv1_out = self.conv1(x2)
        # print(conv1_out.shape)
        conv1_out = conv1_out.view(-1, 1, 2601).view(-1, 1, 51, 51)

        im = conv1_out[0, :].detach().cpu().numpy()
        # plt.imshow(im[0, :])
        # plt.imshow(self.pic[0])
        # plt.imshow(im[0])
        # plt.imsave('out.png', im[0])

        x1 = torch.cat((x1, conv1_out), 1)

        # print(x1.shape)
        x1 = self.conv2d_1(x1)
        # x1 = self.cbam1(x1)
        # print(self.pic.shape)
        # self.pic = x1.detach().cpu().permute(0, 2, 3, 1).numpy()
        x1 = self.residual_block_1(x1)
        # x1 = self.cbam2(x1)
        x1 = self.residual_block_2(x1)
        # x1 = self.cbam3(x1)
        x1 = self.residual_block_3(x1)
        # x1 = self.cbam4(x1)
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)
        out = self.out(x1)
        return out


class resnet18_with_attention_final_without_DM(nn.Module):
    def __init__(self):
        super(resnet18_with_attention_final_without_DM, self).__init__()
        self.pic = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 867, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(867),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)
        # print(x1[0,2,0,0],x2[0,2])

        conv1_out = self.conv1(x2)
        # print(conv1_out.shape)
        conv1_out = conv1_out.view(-1, 1, 2601).view(-1, 1, 51, 51)

        im = conv1_out[0, :].detach().cpu().numpy()
        # plt.imshow(im[0, :])
        # plt.imshow(self.pic[0])
        # plt.imshow(im[0])
        # plt.imsave('out.png', im[0])

        # x1 = torch.cat((x1, conv1_out), 1)
        x1 = conv1_out
        # print(x1.shape)
        x1 = self.conv2d_1(x1)
        x1 = self.cbam1(x1)
        # print(self.pic.shape)
        # self.pic = x1.detach().cpu().permute(0, 2, 3, 1).numpy()
        x1 = self.residual_block_1(x1)
        x1 = self.cbam2(x1)
        x1 = self.residual_block_2(x1)
        x1 = self.cbam3(x1)
        x1 = self.residual_block_3(x1)
        x1 = self.cbam4(x1)
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)
        out = self.out(x1)
        return out
