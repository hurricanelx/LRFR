import os
import time
import sys

import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from PIL import Image

from model.resnet import ResNet34
from src.CosFace import MarginCosineProduct
from src.metrics import ArcMarginProduct
from src.metrics import ExpDist
from src import lfw_eval
from SCface_eval import test_1N_scface
# from src.pair_webface import Pair_webface
from src.pair_mult_solu_webface import Pair_webface_multi
from src.xqlfw_feat import eval_xqlfw
import warnings
warnings.filterwarnings('ignore')


class Logger(object):
    def __init__(self, filename="./log/log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()


torch.backends.cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_interval = 100

test_path1 = '../aligned_SCface_database/SCface/testing_sid_id.lst'
dis1_path = '../aligned_SCface_database/SCface/distance_1_test.lst'
dis2_path = '../aligned_SCface_database/SCface/distance_2_test.lst'
dis3_path = '../aligned_SCface_database/SCface/distance_3_test.lst'

transform_norm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
])
webface = Pair_webface_multi('../../data/webface/imgs', transform_norm, transform_norm, prob=[0.5, 0.5, 1.1])
train_loader = torch.utils.data.DataLoader(webface, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

model = ResNet34()
model_eval = ResNet34().to(device)
model = torch.nn.DataParallel(model).cuda()
print(model)
classifier = MarginCosineProduct(512, 10572, s=48).to(device)
print(classifier)

criterion = torch.nn.CrossEntropyLoss().cuda()
dist_criterion = ExpDist(p=1, device=device).cuda()
optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}],
                            lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [16, 24], gamma=0.1)


def train(train_loader, model, classifier, criterion, dist_criterion, optimizer, log_interval, epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    print("current lr:", optimizer.state_dict()['param_groups'][0]['lr'])
    time_curr = time.time()
    loss_display = 0.0
    loss_cls_display = 0.0
    loss_l2_display = 0.0

    for batch_idx, (img_norm, img_low, target) in enumerate(train_loader, 1):
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        img_norm, img_low, target = img_norm.to(device), img_low.to(device), target.to(device)
        # compute output
        norm_feat = model(img_norm)
        low_feat = model(img_low)
        output_norm = classifier(norm_feat, target)
        output_low = classifier(low_feat, target)

        loss_cls = (criterion(output_norm, target) + criterion(output_low, target))/2
        loss_l2 = dist_criterion(norm_feat, low_feat)
        loss = loss_cls + loss_l2
        loss_cls_display += loss_cls.item()
        loss_l2_display += loss_l2.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= log_interval
            loss_cls_display /= log_interval
            loss_l2_display /= log_interval
            INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(classifier.m, classifier.s)
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss_cls: {:.6f}, Loss_expdist: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(img_norm), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_cls_display, loss_l2_display, time_used, log_interval) + INFO
            )
            time_curr = time.time()
            loss_display = 0.0
            loss_cls_display = 0.0
            loss_l2_display = 0.0
    lr_schedule.step()


def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)


best_lfw_acc = 0
for epoch in range(1, 30):
    train(train_loader, model, classifier, criterion, dist_criterion, optimizer, log_interval, epoch)
    model.module.save('./model_save/' + 'Cos_res34_pair' + str(epoch) + '_checkpoint.pth')
    # torch.save(classifier, './model_save/' + 'CosFace_res34_pair_classifier' + str(epoch) + '_checkpoint.pth')
    acc, _ = lfw_eval.eval(model_eval, './model_save/' + 'Cos_res34_pair' + str(epoch) + '_checkpoint.pth', False)
    if acc > best_lfw_acc:
        best_lfw_acc = acc
    print('best_lfw_acc:', best_lfw_acc)
    total_cmc = test_1N_scface(test_path1, dis1_path, dis2_path, dis3_path, model_eval,
                   './model_save/' + 'Cos_res34_pair' + str(epoch) + '_checkpoint.pth')
    print('mean acc:', (total_cmc[0][0]+total_cmc[1][0]+total_cmc[2][0])/3)
    eval_xqlfw(model_eval, './model_save/' + 'Cos_res34_pair' + str(epoch) + '_checkpoint.pth', stdout=Logger().log)
print('Finished Training')
