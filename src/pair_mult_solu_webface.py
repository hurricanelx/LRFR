import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.utils.data as data
from PIL import Image


class Pair_webface_multi(data.Dataset):
    def __init__(self, root, transform_norm, transform_low, prob=[0.5, 0.8, 1.0], target_trans=None):
        super(Pair_webface_multi, self).__init__()
        self.imagefolder = dataset.ImageFolder(root, target_transform=target_trans)
        self.transform_norm = transform_norm
        self.transform_low = transform_low
        self.trans1 = transforms.Compose([
            transforms.Resize(7, Image.BICUBIC),
            transforms.Resize(112, Image.BICUBIC),
        ])
        self.trans2 = transforms.Compose([
            transforms.Resize(14, Image.BICUBIC),
            transforms.Resize(112, Image.BICUBIC),
        ])
        self.trans3 = transforms.Compose([
            transforms.Resize(20, Image.BICUBIC),
            transforms.Resize(112, Image.BICUBIC),
        ])
        self.trans = [self.trans1, self.trans2, self.trans3]
        self.prob = prob

    def __getitem__(self, index):
        img, target = self.imagefolder.__getitem__(index)
        img_norm = self.transform_norm(img)
        random = torch.rand(1)
        if random <= self.prob[0]:
            img_low = self.trans[0](img)
        elif self.prob[0] < random <= self.prob[1]:
            img_low = self.trans[1](img)
        else:
            img_low = self.trans[2](img)
        img_low = self.transform_low(img_low)
        return img_norm, img_low, target

    def __len__(self):
        return self.imagefolder.__len__()


if __name__ == '__main__':
    transform_norm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    transform_low = transforms.Compose([
        transforms.Resize(16, Image.BICUBIC),
        transforms.Resize(112, Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    webface = Pair_webface_multi('../../data/webface/imgs', transform_norm, transform_low)
    print(len(webface))
