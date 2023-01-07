import os
import cv2
import time
import numpy as np
import sklearn
import torchvision.transforms as transforms
import torch
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import loadmat, savemat
from model.LResNet import LResNet50E_IR
from model.resnet import ResNet34
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import io as scio
import json


def read_pairs_features(pairs_path, features_path, method, feat_norm=True):
    idx2feat = {}
    features = scio.loadmat(os.path.join(features_path, 'features_%s.mat' % method))['features']
    if feat_norm:
        features = preprocessing.normalize(features, norm='l2')
    with open(os.path.join(pairs_path, 'pairs_test.tsv'), 'r') as f:
        pairs = f.read().strip().split('\n')[1:]
        i = 0
        for p in pairs:
            idx2feat[p.strip().split('\t')[1]] = features[i]
            idx2feat[p.strip().split('\t')[17]] = features[i + 1]
            i += 2
        assert i == len(features)
    return idx2feat


def calculate_cmc(gallery_list, distance_1, dis="dis1"):
    gallery_name = []
    gallery_label = []
    for i in range(len(gallery_list)):
        gallery_name.append(gallery_list[i].split("\t")[0])
        gallery_label.append(int(gallery_list[i].split("\t")[1].split("\n")[0]))

    probe_name = []
    probe_label = []
    for i in range(len(distance_1)):
        probe_name.append(distance_1[i].split("\t")[1])
        probe_label.append(int(distance_1[i].split("\t")[-1].split("\n")[0]))

    probe_features = scio.loadmat('result/%s_feature.mat' % dis)[dis + '_feature']
    gallery_features = scio.loadmat('result/frontal_feature.mat')['frontal_feature']

    sim = cosine_similarity(probe_features, gallery_features)  # p x g
    argsim = np.argsort(sim, axis=1)

    ranks = np.zeros(len(gallery_features))
    for i in range(len(probe_features)):  # for the i-th probe
        label = probe_label[i]
        ranking = len(gallery_features) - np.where(argsim[i] == gallery_label.index(label))[0][0]
        ranks += np.concatenate((np.zeros(ranking - 1), np.ones(len(gallery_features) - ranking + 1)), axis=0)
    cmc = ranks / len(probe_features)
    # print cmc
    return cmc


def draw_cmc(total_cmc, save_dir=''):
    # reading cmc of compared methods

    # drawing ROC
    x_labels = [1, 3, 5, 10, 20, 50]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for sset in range(3):
        cmc = total_cmc[sset]
        ax.plot(np.arange(1, 1 + len(cmc)), cmc, label=('[distance %d (Rank-1 : %0.4f %%)]' % (sset, cmc[0] * 100)))

        ax.set_xlim([0, 50])
        ax.set_ylim([0.1, 1.0])
        ax.set_xticks(x_labels)
        ax.set_yticks(np.linspace(0.1, 1.0, 11, endpoint=True))
        ax.set_xlabel('Rank-N')
        ax.set_ylabel('Identification Rate(%)')
        plt.grid(linestyle='--', linewidth=1)
        ax.legend(loc="lower right")

    ax.set_title('CMC on AFW-A three sets')
    fig.savefig(os.path.join(save_dir, 'CMC_%s.pdf' % ("sss")))


class Embedding:
    def __init__(self, model, model_path, ctx_id=0):
        image_size = (112, 112)
        self.image_size = image_size
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.model = model.cuda()
        self.trans = transforms.Compose([
            transforms.Resize(112, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def get(self, img_path):
        img = Image.open(img_path)
        data = self.trans(img).unsqueeze(0).cuda()
        feat = self.model(data).cpu().detach().numpy()
        # print(feat.shape)
        feat = feat.reshape([-1, feat.shape[0] * feat.shape[1]])
        feat = sklearn.preprocessing.normalize(feat).flatten()
        return feat


def get_image_feature(img_path, img_list_path, model, model_path):
    embedding = Embedding(model, model_path)
    files = img_list_path
    img_feats = []

    for img_index in range(len(files)):
        img_name = os.path.join(img_path, img_list_path[img_index])
        # img = cv2.imread(img_name)
        # try:
        #     img_feats.append(embedding.get(img))
        # except:
        #     print(img_name)
        #     print(img_index)
        img_feats.append(embedding.get(img_name))
        if img_index % 500 == 0:
            print(img_index, len(files), time.ctime())

    img_feats = np.array(img_feats).astype(np.float32)
    return img_feats


def gen_SCface_feat(test_path1, dis1_path, dis2_path, dis3_path, model, model_path):
    print("\nStep1: Load Meta Data")
    frontal = open(test_path1).readlines()
    dis1 = open(dis1_path).readlines()
    dis2 = open(dis2_path).readlines()
    dis3 = open(dis3_path).readlines()

    frontal_img = []
    for i in range(len(frontal)):
        data = frontal[i].split("\t")[0]
        data = data.replace('fang', 'lx')
        frontal_img.append(data)

    dis1_img = []
    for i in range(len(dis1)):
        data = dis1[i].split("\t")[1]
        data = data.replace('fang', 'lx')
        dis1_img.append(data)

    dis2_img = []
    for i in range(len(dis2)):
        data = dis2[i].split("\t")[1]
        data = data.replace('fang', 'lx')
        dis2_img.append(data)
    dis3_img = []
    for i in range(len(dis3)):
        data = dis3[i].split("\t")[1]
        data = data.replace('fang', 'lx')
        dis3_img.append(data)

    image_dir = ''
    result_dir = 'result/'

    frontal_feature = get_image_feature(image_dir, frontal_img, model, model_path)
    dis1_feature = get_image_feature(image_dir, dis1_img, model, model_path)
    dis2_feature = get_image_feature(image_dir, dis2_img, model, model_path)
    dis3_feature = get_image_feature(image_dir, dis3_img, model, model_path)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    savemat(result_dir + "frontal_feature.mat", {'frontal_feature': frontal_feature})
    savemat(result_dir + "dis1_feature.mat", {'dis1_feature': dis1_feature})
    savemat(result_dir + "dis2_feature.mat", {'dis2_feature': dis2_feature})
    savemat(result_dir + "dis3_feature.mat", {'dis3_feature': dis3_feature})


def test_1N_scface(test_path1, dis1_path, dis2_path, dis3_path, model, model_path):
    gen_SCface_feat(test_path1, dis1_path, dis2_path, dis3_path, model, model_path)
    gallery_list = open(test_path1).readlines()
    distance_1 = open(dis1_path).readlines()
    distance_2 = open(dis2_path).readlines()
    distance_3 = open(dis3_path).readlines()

    print('reading features of pairs...')

    total_cmc = []
    cmc = calculate_cmc(gallery_list, distance_1, "dis1")
    print("distance 1 rank1: %f" % cmc[0])
    total_cmc.append(cmc)
    cmc = calculate_cmc(gallery_list, distance_2, "dis2")
    print("distance 2 rank1: %f" % cmc[0])

    total_cmc.append(cmc)
    cmc = calculate_cmc(gallery_list, distance_3, "dis3")
    print("distance 3 rank1: %f" % cmc[0])

    total_cmc.append(cmc)
    return total_cmc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    test_path1 = '/home/pris/lx/aligned_SCface_database/SCface/testing_sid_id.lst'
    dis1_path = '/home/pris/lx/aligned_SCface_database/SCface/distance_1_test.lst'
    dis2_path = '/home/pris/lx/aligned_SCface_database/SCface/distance_2_test.lst'
    dis3_path = '/home/pris/lx/aligned_SCface_database/SCface/distance_3_test.lst'
    model = ResNet34()
    # model_path = '/home/pris/lx/LRFR/model_save/CosFace_1_checkpoint.pth'
    model_path = '/home/pris/lx/LRFR/model_save/CosFace_res34_pair1_checkpoint.pth'
    test_1N_scface(test_path1, dis1_path, dis2_path, dis3_path, model, model_path)
