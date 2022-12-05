from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import torch
import numpy as np


# 数据集
class myDataSet(Dataset):
    def __init__(self, train=True, vaild=False, extra=False):
        self.train = train
        self.vaild = vaild
        self.extra = extra
        dataset = []
        data_pics = None
        if train:
            with open('dataset1.txt') as f:
                for line in f.readlines():
                    data = [float(i) for i in line.split()]
                    # if data[2] == 75 or data[2] == 80:
                    #    continue
                    dataset.append(data)
            dataset = np.array(dataset)
            data_pics = np.zeros(shape=[17, 3, 51, 51])
            i = 0
            cnt = 0
            for data in dataset:

                cnt += 1
                data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                if cnt == 2601:
                    cnt = 0
                    i += 1
            print(data_pics.shape)
            self.data_pics = data_pics

            np.random.shuffle(dataset)
            self.data = dataset

            del dataset, data_pics
        else:
            if vaild:
                with open('dataset1.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        # if data[2] == 75 or data[2] == 80:
                        #    continue
                        dataset.append(data)
                dataset = np.array(dataset)
                data_pics = np.zeros(shape=[17, 3, 51, 51])
                i = 0
                cnt = 0
                for data in dataset:

                    cnt += 1
                    data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                    if cnt == 2601:
                        cnt = 0
                        i += 1
                self.data_pics = data_pics
                dataset = []
                with open('dataset_val.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        if data[2] == 47.5 and not self.extra:
                            break
                        dataset.append(data)
                dataset = np.array(dataset)
                self.data = dataset
            else:

                with open('dataset_val.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        if data[2] == 47.5 and not self.extra:
                            break
                        dataset.append(data)
                dataset = np.array(dataset)
                data_pics = np.zeros(shape=[17, 3, 51, 51])
                i = 0
                cnt = 0
                for data in dataset:
                    cnt += 1
                    data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                    if cnt == 2601:
                        cnt = 0
                        i += 1
                print(data_pics.shape)
                self.data_pics = data_pics
                self.data = dataset
            del dataset, data_pics

    def __getitem__(self, index):

        if self.train:

            # 循环遍历17张数据图
            for i in range(17):

                if self.data_pics[i, -1, 0, 0] == self.data[index][2]:

                    return torch.FloatTensor(self.data_pics[i]), \
                        torch.FloatTensor(self.data[index][:3]), \
                        torch.FloatTensor(self.data[index][3:5])
        else:
            if self.vaild:
                # 将角度映射到存在数据图的角度
                angle = self.data[index][2]
                angle_low = (angle // 5) * 5
                angle_high = angle_low + 5
                for i in range(17):
                    if self.data_pics[i, -1, 0, 0] == angle_low:
                        pic_low = self.data_pics[i]
                    if self.data_pics[i, -1, 0, 0] == angle_high:
                        pic_high = self.data_pics[i]
                return torch.FloatTensor(((pic_high - pic_low) / 5) * (angle - angle_low) + pic_low),\
                    torch.FloatTensor(self.data[index][:3]),\
                    torch.FloatTensor(self.data[index][3:5])
                # if 5 - (angle - (angle // 5) * 5) < 2.5:
                #     angle = (angle // 5) * 5
                # else:
                #     angle = (angle // 5 + 1) * 5
                # print('使用角度为' + str(angle) + '的数据图')
                # 循环遍历17张数据图
                # for i in range(17):
                #     if self.data_pics[i, -1, 0, 0] == angle:
                #         return torch.FloatTensor(self.data_pics[i]), torch.FloatTensor(self.data[index][:3]), torch.FloatTensor(self.data[index][3:5])
            else:
                # 循环遍历数据图
                for i in range(self.data_pics.shape[0]):
                    if self.data_pics[i, -1, 0, 0] == self.data[index][2]:
                        return torch.FloatTensor(self.data_pics[i]), torch.FloatTensor(self.data[index][:3]), torch.FloatTensor(self.data[index][3:5])

    def __len__(self):
        return len(self.data)


class myDataSet_u(Dataset):
    def __init__(self, train=True, vaild=False, extra=False):
        self.train = train
        self.vaild = vaild
        self.extra = extra
        dataset = []
        data_pics = None
        if train:
            with open('dataset1.txt') as f:
                for line in f.readlines():
                    data = [float(i) for i in line.split()]
                    # if data[2] == 75 or data[2] == 80:
                    #    continue
                    dataset.append(data)
            dataset = np.array(dataset)
            data_pics = np.zeros(shape=[17, 3, 51, 51])

            i = 0
            cnt = 0
            for data in dataset:

                cnt += 1
                data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                if cnt == 2601:
                    cnt = 0
                    i += 1
            print(data_pics.shape)
            self.data_pics = data_pics

            np.random.shuffle(dataset)
            self.data = dataset

            del dataset, data_pics
        else:
            if vaild:
                with open('dataset1.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        # if data[2] == 75 or data[2] == 80:
                        #    continue
                        dataset.append(data)
                dataset = np.array(dataset)
                data_pics = np.zeros(shape=[17, 3, 51, 51])
                i = 0
                cnt = 0
                for data in dataset:

                    cnt += 1
                    data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                    if cnt == 2601:
                        cnt = 0
                        i += 1
                self.data_pics = data_pics
                dataset = []
                with open('dataset_val.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        if data[2] == 47.5 and not self.extra:
                            break
                        dataset.append(data)
                dataset = np.array(dataset)
                self.data = dataset
            else:

                with open('dataset_val.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        if data[2] == 47.5 and not self.extra:
                            break
                        dataset.append(data)
                dataset = np.array(dataset)
                data_pics = np.zeros(shape=[17, 3, 51, 51])
                i = 0
                cnt = 0
                for data in dataset:
                    cnt += 1
                    data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                    if cnt == 2601:
                        cnt = 0
                        i += 1
                print(data_pics.shape)
                self.data_pics = data_pics
                self.data = dataset
            del dataset, data_pics

    def __getitem__(self, index):

        out = np.zeros(shape=[51, 51])

        if self.train:

            # 循环遍历17张数据图
            for i in range(17):

                if self.data_pics[i, -1, 0, 0] == self.data[index][2]:
                    out[int(self.data[index, 3]), int(self.data[index, 4])] = 10

                    return torch.FloatTensor(self.data_pics[i]), torch.FloatTensor(self.data[index][:3]), torch.FloatTensor(out)
        else:
            if self.vaild:
                # 将角度映射到存在数据图的角度
                angle = self.data[index][2]
                if 5 - (angle - (angle // 5) * 5) < 2.5:
                    angle = (angle // 5) * 5
                else:
                    angle = (angle // 5 + 1) * 5
                # print('使用角度为' + str(angle) + '的数据图')
                # 循环遍历17张数据图
                for i in range(17):
                    if self.data_pics[i, -1, 0, 0] == angle:
                        out[int(self.data[index, 3]), int(self.data[index, 4])] = 10

                        return torch.FloatTensor(self.data_pics[i]), torch.FloatTensor(self.data[index][:3]), torch.FloatTensor(out)
            else:
                # 循环遍历数据图
                for i in range(self.data_pics.shape[0]):
                    if self.data_pics[i, -1, 0, 0] == self.data[index][2]:
                        out[int(self.data[index, 3]), int(self.data[index, 4])] = 10

                        return torch.FloatTensor(self.data_pics[i]), torch.FloatTensor(self.data[index][:3]), torch.FloatTensor(out)

    def __len__(self):
        return len(self.data)
