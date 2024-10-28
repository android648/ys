import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
from skimage.measure import label
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from networks.vnet import VNet
# from networks.vnet_DWC import VNet
from networks.vnet_duohe import VNet
from networks.ResNet34 import Resnet34
# from networks.ResNet50 import Resnet34
# from networks.ResNet18 import Resnet34
from utils import ramps, losses, test_3d_patch
from utils.mix_module import mix_module, pseudo_label
from dataloaders.la_heart import BraTS, Pancras, LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler
from utils.Cut_utils import cutmix_mask, mix_loss, parameter_sharing, update_ema_variables

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/datasets/Pancreas', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--root_path_list', type=str, default='/root/autodl-tmp/datasets/Pancreas/', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--exp', type=str,  default="Pan_12_VDWC", help='model_name')                               # todo model name
parser.add_argument('--max_iterations', type=int,  default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--type', type=str,  default='Pan', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')

args = parser.parse_args()

train_data_path = args.root_path
train_data_path1 = args.root_path_list
snapshot_path = "/root/autodl-tmp/ys/cutmix/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
T = 0.1

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c-1):
        temp_line = vec[:,i,:].unsqueeze(1)  # b 1 c
        star_index = i+1
        rep_num = c-star_index
        repeat_line = temp_line.repeat(1, rep_num,1)
        two_patch = vec[:,star_index:,:]
        temp_cat = torch.cat((repeat_line,two_patch),dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result,dim=1)
    return  result

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name ='vnet'):
        # Network definition
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        return model

    model_vnet = create_model(name='vnet')
    model_resnet = create_model(name='resnet34')

    if args.type == 'LA':
        patch_size = (112, 112, 80)
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           train_flod='train0.list',                   # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                           ]),
                           sp_transform=transforms.Compose([
                               ToTensor(),
                           ]))

        labeled_idxs = list(range(16))           # todo set labeled num
        unlabeled_idxs = list(range(16, 80))     # todo set labeled num all_sample_num

    elif args.type == 'Pan' :
        patch_size = (96, 96, 96)
        db_train = Pancras(base_dir=train_data_path,base_dir_list=train_data_path1,
                           split='train',
                           train_flod='train0.list',  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                           ]),
                           sp_transform=transforms.Compose([
                               ToTensor(),
                           ]))

        labeled_idxs = list(range(12))  # todo set labeled num
        unlabeled_idxs = list(range(12, 62))  # todo set labeled num all_sample_num

    else:
        patch_size = (96, 96, 96)
        db_train = BraTS(base_dir=train_data_path, base_dir_list=train_data_path1,
                         split='train',
                         train_flod='train.txt',  # todo change training flod
                         common_transform=transforms.Compose([
                             RandomCrop(patch_size),
                         ]),
                         sp_transform=transforms.Compose([
                             ToTensor(),
                         ]))
        labeled_idxs = list(range(50))  # todo set labeled num
        unlabeled_idxs = list(range(50, 250))  # todo set labeled num all_sample_num

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice=0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model_vnet.train()
    model_resnet.train()

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            W = iter_num / max_iterations
            time2 = time.time()
            print('epoch:{},i_batch:{}'.format(epoch_num,i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']

            v_img, v_label = volume_batch1[:labeled_bs].cuda(), volume_label1[:labeled_bs].cuda()
            uv_img = volume_batch1[labeled_bs:].cuda()
            r_img, r_label = volume_batch2[:labeled_bs].cuda(), volume_label2[:labeled_bs].cuda()
            ur_img = volume_batch2[labeled_bs:].cuda()
            plr, _ = model_vnet(ur_img)
            plv, _ = model_resnet(uv_img)
            pl_v = get_cut_mask(plv, nms=1)
            pl_r = get_cut_mask(plr, nms=1)
            img_mask1, loss_mask1 = cutmix_mask(v_img, 4)
            img_mask2, loss_mask2 = cutmix_mask(r_img, 4)

            v_mix_img1 = v_img * img_mask1 + uv_img * (1 - img_mask1)
            r_mix_img2 = r_img * img_mask2 + ur_img * (1 - img_mask2)

            v_input = torch.cat((v_img, uv_img, v_mix_img1), 0)
            r_input = torch.cat((r_img, ur_img, r_mix_img2), 0)

            for num in range(3):
                v_outputs, stage_v = model_vnet(v_input)
                r_outputs, stage_r = model_resnet(r_input)

                for idx in range(len(stage_v)):
                    maskv = stage_v[idx].detach()
                    maskr = stage_r[idx].detach()

                    v_input = v_input + 1e-3*(maskr-maskv)
                    r_input = r_input + 1e-3*(maskv-maskr)

                v_loss_seg = F.cross_entropy(v_outputs[:labeled_bs], v_label[:labeled_bs])
                v_outputs_soft = F.softmax(v_outputs, dim=1)
                v_loss_dice = losses.dice_loss(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] == 1)

                r_loss_seg = F.cross_entropy(r_outputs[:labeled_bs], r_label[:labeled_bs])
                r_outputs_soft = F.softmax(r_outputs, dim=1)
                r_loss_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] == 1)

                v_supervised_loss = v_loss_seg + v_loss_dice
                r_supervised_loss = r_loss_seg + r_loss_dice
                ## calculate the supervised loss
                v_outputs_clone = v_outputs_soft[labeled_bs:batch_size, :, :, :, :].clone().detach()
                r_outputs_clone = r_outputs_soft[labeled_bs:batch_size, :, :, :, :].clone().detach()
                v_outputs_clone1 = torch.pow(v_outputs_clone, 1 / T)
                r_outputs_clone1 = torch.pow(r_outputs_clone, 1 / T)
                v_outputs_clone2 = torch.sum(v_outputs_clone1, dim=1, keepdim=True)
                r_outputs_clone2 = torch.sum(r_outputs_clone1, dim=1, keepdim=True)
                v_outputs_PLable = torch.div(v_outputs_clone1, v_outputs_clone2)
                r_outputs_PLable = torch.div(r_outputs_clone1, r_outputs_clone2)

                consistency_weight = get_current_consistency_weight(iter_num // 150)
                r_consistency_dist = consistency_criterion(r_outputs_soft[labeled_bs:batch_size, :, :, :, :], v_outputs_PLable)
                b, c, w, h, d = r_consistency_dist.shape
                r_consistency_dist = torch.sum(r_consistency_dist) / (b * c * w * h * d)
                r_consistency_loss = r_consistency_dist
                writer.add_scalar('loss/r_consistency_loss', r_consistency_loss, iter_num)

                v_consistency_dist = consistency_criterion(v_outputs_soft[labeled_bs:batch_size, :, :, :, :], r_outputs_PLable)
                b, c, w, h, d = v_consistency_dist.shape
                v_consistency_dist = torch.sum(v_consistency_dist) / (b * c * w * h * d)
                v_consistency_loss = v_consistency_dist
                writer.add_scalar('loss/v_consistency_loss', v_consistency_loss, iter_num)

                v_loss_mix1 = mix_loss(v_outputs[2*labeled_bs:], v_label, pl_v, loss_mask1, u_weight=args.u_weight)
                r_loss_mix2 = mix_loss(r_outputs[2*labeled_bs:], r_label, pl_r, loss_mask2, u_weight=args.u_weight)

                v_loss = v_supervised_loss + v_loss_mix1 + consistency_weight * v_consistency_loss
                r_loss = r_supervised_loss + r_loss_mix2 + consistency_weight * r_consistency_loss

                if (torch.any(torch.isnan(v_loss)) or torch.any(torch.isnan(r_loss))):
                    print('nan find')
                vnet_optimizer.zero_grad()
                resnet_optimizer.zero_grad()
                v_loss.backward()
                r_loss.backward()
                vnet_optimizer.step()
                resnet_optimizer.step()
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/v_loss', v_loss, iter_num)
                writer.add_scalar('loss/r_loss', r_loss, iter_num)
                writer.add_scalar('loss/v_loss_seg', v_loss_seg, iter_num)
                writer.add_scalar('loss/v_loss_dice', v_loss_dice, iter_num)
                writer.add_scalar('loss/r_loss_seg', r_loss_seg, iter_num)
                writer.add_scalar('loss/r_loss_dice', r_loss_dice, iter_num)
                writer.add_scalar('loss/v_loss_mix1', v_loss_mix1, iter_num)
                writer.add_scalar('loss/r_loss_mix2', r_loss_mix2, iter_num)

            logging.info(
                'iteration ： %d v_supervised_loss : %f v_loss_mix1 :%f v_consistency_loss : %f r_supervised_losss : %f r_loss_mix2 :  %f r_consistency_loss : %f'  %
                (iter_num,
                 v_supervised_loss.item(), v_loss_mix1.item(), v_consistency_loss.item(), r_supervised_loss.item(), r_loss_mix2.item(), r_consistency_loss.item()))

            ## change lr
            if iter_num % 6000 == 0 and iter_num!= 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num > 0 and iter_num % 200 == 0:
                model_vnet.eval()
                model_resnet.eval()
                if args.type == "LA":
                    dice_sample = test_3d_patch.var_all_case(model_vnet,model_resnet, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4)
                elif args.type == "Pan":
                    dice_sample = test_3d_patch.var_all_case(model_vnet,model_resnet, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas_CT')
                else:
                    dice_sample = test_3d_patch.var_all_case(model_vnet,model_resnet, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=64, stride_z=64, dataset_name='Bra')
                if dice_sample > best_dice:
                    best_dice = dice_sample

                    save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_{}'.format(best_dice) + '_{}'.format(iter_num) + '.pth')
                    torch.save(model_vnet.state_dict(), save_mode_path_vnet)
                    logging.info("save model to {}".format(save_mode_path_vnet))

                    save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_{}'.format(best_dice) + '_{}'.format(iter_num) + '.pth')
                    torch.save(model_resnet.state_dict(), save_mode_path_resnet)
                    logging.info("save model to {}".format(save_mode_path_resnet))

                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model_vnet.train()
                model_resnet.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break


    writer.close()
