import os
import argparse
import torch
# from networks.vnet import VNet
# from networks.vnet_duohe import VNet
from networks.vnet_DWC import VNet
from networks.ResNet34 import Resnet34
# from networks.ResNet18 import Resnet34
from test_util import test_all_case
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,  default="Bra_50_VDWC", help='model_name')                # todo change test model name
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--type', type=str,  default='Bra', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "/root/autodl-tmp/ys/cutmix/"+FLAGS.model+'/'
test_save_path = "/root/autodl-tmp/ys/cutmix/"+FLAGS.model+'/'+"zhu/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
num_classes = 2
image_lists=[]
def create_model(name='vnet'):
    # Network definition
    if name == 'vnet':
        # net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
    if name == 'resnet34':
        net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        # net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()

    return model

def test_calculate_metric(epoch_num):
    vnet   = create_model(name='vnet')
    resnet = create_model(name='resnet34')

    v_save_mode_path = os.path.join(snapshot_path, 'vnet_iter_' + str(epoch_num) + '.pth')
    vnet.load_state_dict(torch.load(v_save_mode_path))
    print("init weight from {}".format(v_save_mode_path))
    vnet.eval()

    r_save_mode_path = os.path.join(snapshot_path, 'resnet_iter_' + str(epoch_num) + '.pth')
    resnet.load_state_dict(torch.load(r_save_mode_path))
    print("init weight from {}".format(r_save_mode_path))
    resnet.eval()

    if FLAGS.type == "LA":
        patch_size = (112, 112, 80)
        avg_metric = test_all_case(vnet, resnet, num_classes=num_classes,
                                                 patch_size=patch_size,
                                                 stride_xy=18, stride_z=4, dataset_name='LA', test_save_path=test_save_path)
    elif FLAGS.type == "Pan":
        patch_size = (96, 96, 96)
        avg_metric = test_all_case(vnet, resnet, num_classes=num_classes,
                                                 patch_size=patch_size,
                                                 stride_xy=16, stride_z=16, dataset_name='Pancreas_CT', test_save_path=test_save_path)
    elif FLAGS.type == 'Bra':
        patch_size = (96, 96, 96)
        avg_metric = test_all_case(vnet, resnet, num_classes=num_classes,
                                                 patch_size=patch_size,
                                                 stride_xy=64, stride_z=64, dataset_name='Bra', test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    iters = '20'
    metric = test_calculate_metric(iters)
    print('iter:', iter)
    print(metric)
