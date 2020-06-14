import torch
import torch.utils.data
import pickle
import numpy
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as F


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, train = True):
        self.train = train
        # with open('D:\\Tsinghua\\研究生课程\\模式识别\\experiments\\exp2\\ResNet_18_cifar10\\cifar','rb') as fo:
        with open('cifar','rb') as fo:
            cifar = pickle.load(fo, encoding='bytes')
        if self.train:
            self.train_data = cifar['train_data']
            self.train_labels = cifar['train_labels']
            # print(numpy.min(self.train_labels))
            self.train_data = self.train_data.reshape(-1, 3, 32, 32)
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            # Pre-process for dataset
            self.transform_train_pre = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # padding and randomcrop to 32*32
                transforms.RandomHorizontalFlip(),  # flip the image horizontally with a probability of 0.5
                # transforms.RandomVerticalFlip()
            ])

            self.transform_train_post = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # mean and variance for R,G,B normalization
            ])
        else:
            self.test_data = cifar['test_data']
            self.test_labels = cifar['test_labels']
            self.test_data = self.test_data.reshape(-1, 3, 32, 32)
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        # print(self.train_data.shape)
        # print(self.train_labels.shape)
        # print(isinstance(self.train_data, numpy.ndarray))

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        # img pre-process
        if self.train:
            img = self.transform_train_pre(img)
            # img = self.my_rotate(img)
            img = self.transform_train_post(img)
        else:
            img = self.transform_test(img)
        # label process
        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    # transform labels to torch.LongTensor type
    def target_transform(self, label):
        label = numpy.array(label)
        target = torch.from_numpy(label).long()
        return target

    def my_rotate(self, img):
        aa = random.uniform(0, 1)

        if aa < 0.7:
            img = img
        elif aa < 0.78:
            img = F.rotate(img, 30, resample=Image.BILINEAR, expand=False, center=None)
        elif aa < 0.86:
            img = F.rotate(img, 330, resample=Image.BILINEAR, expand=False, center=None)

        elif aa < 0.91:
            img = F.rotate(img, 60, resample=Image.BILINEAR, expand=False, center=None)
        elif aa < 0.96:
            img = F.rotate(img, 300, resample=Image.BILINEAR, expand=False, center=None)

        elif aa < 0.98:
            img = F.rotate(img, 90, resample=Image.BILINEAR, expand=False, center=None)
        else:
            img = F.rotate(img, 270, resample=Image.BILINEAR, expand=False, center=None)

        return img

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

if __name__ == '__main__':
    trainset = CIFAR10(train= True)
    testset = CIFAR10(train= False)
    # img, label = trainset.__getitem__(233)