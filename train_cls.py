from ModelNetDataLoader import ModelNetDataset
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from torch.utils.data import DataLoader
from PointNet2.pointnet2_cls_ssg import get_model,get_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet++')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    #parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument("--dataset_type", type=str, default="modelnet40", help="dataset type modelnet40")
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc




if __name__ == '__main__':
        opt = parse_args()
        print(opt)


        if opt.dataset_type == "modelnet40":
            dataset = ModelNetDataset(
                root="E:/NewPython/PointNet/ModelNet40/",
                npoints=opt.num_point,
                split="train"
            )

            test_dataset = ModelNetDataset(
                root="E:/NewPython/PointNet/ModelNet40/",
                split="test",
                npoints=opt.num_point,
                data_augmentation=False
            )

        else:
            exit("wrong dataset type")

        traindataloader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4
        )
        print(len(dataset))

        testdataloader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4
        )
        print(len(test_dataset))

        print(len(dataset), len(test_dataset))
        num_classes = len(dataset.classes)
        print('classes', num_classes)

        classifier = get_model(num_classes, normal_channel=opt.normal).cuda()
        criterion = get_loss().cuda()

        try:

            classifier.load_state_dict(torch.load['model_state_dict'])
            print('Use pretrain model')
        except:
            print('No existing model, starting training from scratch...')
            start_epoch = 0

        if opt.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=opt.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=opt.decay_rate
            )
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        global_epoch = 0
        global_step = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0
        mean_correct = []

        '''TRANING'''
        print('Start training...')
        for epoch in range(start_epoch, opt.epoch):
            print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, opt.epoch))

            scheduler.step()
            for batch_id, data in tqdm(enumerate(traindataloader, 0), total=len(traindataloader), smoothing=0.9):
                points, target = data
                points = points.data.numpy()
                points = provider.random_point_dropout(points)
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                target = target[:, 0]

                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                optimizer.zero_grad()

                classifier = classifier.train()
                pred, trans_feat = classifier(points)
                loss = criterion(pred, target.long(), trans_feat)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))
                loss.backward()
                optimizer.step()
                global_step += 1

            train_instance_acc = np.mean(mean_correct)
            print('Train Instance Accuracy: %f' % train_instance_acc)

            with torch.no_grad():
                instance_acc, class_acc = test(classifier.eval(), testdataloader)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc
                print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

                if (instance_acc >= best_instance_acc):
                    print('Save model...')
                    savepath = '/best_model.pth'
                    print('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                global_epoch += 1

        print('End of training...')