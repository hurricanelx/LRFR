from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
import os
import subprocess
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
cudnn.benchmark = True


def extractDeepFeature(img, model, is_gray):
    if is_gray:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(112),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    img, img_ = transform(img), transform(F.hflip(img))
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu').detach().numpy()
    return ft


def eval_xqlfw(model, model_path=None, is_gray=False, stdout=None):
    predicts = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    root = '../data/xqlfw_aligned_112/'
    with open('../data/xqlfw_aligned_112/xqlfw_pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    features = []
    labels = []
    with torch.no_grad():
        for i in tqdm(range(6000)):
            p = pairs_lines[i].replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = True
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = False
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")

            with open(root + name1, 'rb') as f:
                img1 = Image.open(f).convert('RGB')
            with open(root + name2, 'rb') as f:
                img2 = Image.open(f).convert('RGB')
            f1 = extractDeepFeature(img1, model, is_gray)
            f2 = extractDeepFeature(img2, model, is_gray)
            features.append(f1)
            features.append(f2)
            labels.append(sameflag)
            # distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            # predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))
    arr = [features, labels]
    np.save('./result/xqlfw.npy', arr)
    ret = subprocess.run(['python3', './src/xqlfw_eval.py', '--source_embeddings=./result/', '--save=False'], stdout=stdout)
