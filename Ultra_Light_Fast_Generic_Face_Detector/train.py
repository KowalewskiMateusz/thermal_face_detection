"""
This code is the main training code.
"""
import argparse
import itertools
import logging
import os
import sys

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset

from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config.fd_config import define_img_size
from vision.utils.misc import str2bool, freeze_net_layers, store_labels

parser = argparse.ArgumentParser(description='train With Pytorch')

# Params for SGD
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-2,
                    type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4,
                    type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1,
                    type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr',
                    default=None,
                    type=float,
                    help='initial learning rate for base net.')
parser.add_argument(
    '--extra_layers_lr',
    default=None,
    type=float,
    help=
    'initial learning rate for the layers not in base net and prediction heads.'
)

# Params for loading pretrained basenet or checkpoints.

parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument(
    '--scheduler',
    default="multi-step",
    type=str,
    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones',
                    default="80,100",
                    type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max',
                    default=120,
                    type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size',
                    default=24,
                    type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs',
                    default=200,
                    type=int,
                    help='the number epochs')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs',
                    default=5,
                    type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps',
                    default=100,
                    type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda',
                    default=True,
                    type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder',
                    default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument(
    '--log_dir',
    default='./models/Ultra-Light(1MB)_&_Fast_Face_Detector/logs',
    help='lod dir')
parser.add_argument(
    '--cuda_index',
    default="0",
    type=str,
    help='Choose cuda index.If you have 4 GPUs, you can set it like 0,1,2,3')
parser.add_argument('--power', default=2, type=int, help='poly lr pow')
parser.add_argument('--overlap_threshold',
                    default=0.35,
                    type=float,
                    help='overlap_threshold')
parser.add_argument('--optimizer_type',
                    default="SGD",
                    type=str,
                    help='optimizer_type')
parser.add_argument(
    '--input_size',
    default=320,
    type=int,
    help=
    'define network input size,default optional value 128/160/320/480/640/1280'
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

train_datasets = 'data/images/eti'
validation_dataset = 'data/images/koronawirus_A320'
neg_dataset = ''
freeze_all = False
freeze_base = True
resume = 'models/pretrained/version-RFB-320.pth'

logging.info("inpu size :{}".format(args.input_size))
define_img_size(args.input_size)

from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.ssd import MatchPrior

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def lr_poly(base_lr, iter):
    return base_lr * ((1 - float(iter) / args.num_epochs)**(args.power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.lr, i_iter)
    optimizer.param_groups[0]['lr'] = lr


def train(loader,
          net,
          criterion,
          optimizer,
          device,
          debug_steps=100,
          epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        print(".", end="", flush=True)
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(
            confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            print(".", flush=True)
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(f"Epoch: {epoch}, Step: {i}, " +
                         f"Average Loss: {avg_loss:.4f}, " +
                         f"Average Regression Loss {avg_reg_loss:.4f}, " +
                         f"Average Classification Loss: {avg_clf_loss:.4f}")

            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(
                confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':

    create_net = create_Mb_Tiny_RFB_fd
    config = fd_config

    train_transform = TrainAugmentation(config.image_size, config.image_mean,
                                        config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, args.overlap_threshold)
    test_transform = TestTransform(config.image_size, config.image_mean_test,
                                   config.image_std)

    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    logging.info("Prepare training datasets.")

    datasets = list()

    train_dataset = VOCDataset(train_datasets,
                               transform=train_transform,
                               target_transform=target_transform)
    label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
    num_classes = len(train_dataset.class_dict)
    datasets.append(train_dataset)

    # train_dataset = VOCDataset(neg_dataset,pos=False,transform=train_transform,
    # target_transform=target_transform)
    # train_dataset = ConcatDataset(datasets)

    train_loader = DataLoader(train_dataset,
                              args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_dataset = VOCDataset(validation_dataset,
                             transform=test_transform,
                             target_transform=target_transform,
                             is_test=True)

    val_loader = DataLoader(val_dataset,
                            args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    logging.info("Build network.")
    net = create_net(num_classes)

    if torch.cuda.device_count() >= 1:
        cuda_index_list = [int(v.strip()) for v in args.cuda_index.split(",")]
        net = nn.DataParallel(net, device_ids=cuda_index_list)
        logging.info("use gpu :{}".format(cuda_index_list))

    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr

    if freeze_base:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(),
                                 net.extras.parameters(),
                                 net.regression_headers.parameters(),
                                 net.classification_headers.parameters())
        params = [{
            'params':
            itertools.chain(net.source_layer_add_ons.parameters(),
                            net.extras.parameters()),
            'lr':
            extra_layers_lr
        }, {
            'params':
            itertools.chain(net.regression_headers.parameters(),
                            net.classification_headers.parameters())
        }]
    elif freeze_all:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(),
                                 net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")

    # net.init_from_base_net(resume)
    net.init_from_pretrained_ssd(resume)
    # net.load(resume)

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors,
                             neg_pos_ratio=1,
                             center_variance=0.1,
                             size_variance=0.2,
                             device=DEVICE)

    optimizer = torch.optim.Adam(params, lr=args.lr)

    logging.info(
        f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, " +
        f"Extra Layers learning rate: {extra_layers_lr}.")

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):

        train(train_loader,
              net,
              criterion,
              optimizer,
              device=DEVICE,
              debug_steps=args.debug_steps,
              epoch=epoch)

        logging.info("lr rate :{}".format(optimizer.param_groups[0]['lr']))

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            logging.info("lr rate :{}".format(optimizer.param_groups[0]['lr']))
            val_loss, val_regression_loss, val_classification_loss = test(
                val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " + f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder,
                                      f"Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
