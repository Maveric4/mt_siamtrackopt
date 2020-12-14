"""
PyTorch implementation of SiamFC (Luca Bertinetto, et al., ECCVW, 2016)
Written by Heng Fan
"""

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from datetime import datetime as dt
from FINN.SiamNet_finn import *
from Train.VIDDataset import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from Train.Config import *
from Train.Utils import *
import torchvision.transforms as transforms
from Train.DataAugmentation import *
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# fix random seed
np.random.seed(1357)
torch.manual_seed(1234)
from torch.utils.tensorboard import SummaryWriter


def train(data_dir, train_imdb, val_imdb, model_save_path="./model/", use_gpu=True, type=None, config=None):
    # initialize training configuration
    config = config

    # do data augmentation in PyTorch;
    # you can also do complex data augmentation as in the original paper
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride

    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # load data (see details in VIDDataset.py)
    train_dataset = VIDDataset(train_imdb, data_dir, config, train_z_transforms, train_x_transforms)
    val_dataset = VIDDataset(val_imdb, data_dir, config, valid_z_transforms, valid_x_transforms, "Validation")

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.train_num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.val_num_workers, drop_last=True)

    # create SiamFC network architecture (see details in SiamNet.py)
    # if type == NetType.BINARY:
    #     net = binarySiamese.siam()
    # else:
    net = SiamNet(network_type=type)

    # define training strategy;
    # the learning rate of adjust layer (i.e., a conv layer)
    # is set to 0 as in the original paper
    optimizer = torch.optim.SGD([
        {'params': net.conv_features.parameters()},
        {'params': net.adjust.bias},
        {'params': net.adjust.weight, 'lr': 0.0},
    ], config.lr, config.momentum, weight_decay=config.weight_decay)

    # move network to GPU if using GPU
    if use_gpu:
        net.cuda()

    # adjusting learning in each epoch
    scheduler = StepLR(optimizer, config.step_size, config.gamma)

    # used to control generating label for training;
    # once generated, they are fixed since the labels for each
    # pair of images (examplar z and search region x) are the same
    train_response_flag = False
    valid_response_flag = False

    # ------------------------ training & validation process ------------------------
    for i in range(config.num_epoch):

        # adjusting learning rate
        scheduler.step()

        # ------------------------------ training ------------------------------
        # indicating training (very important for batch normalization)
        net.train()

        # used to collect loss
        train_loss = []

        for j, data in enumerate(tqdm(train_loader)):

            # fetch data, i.e., B x C x W x H (batchsize x channel x wdith x heigh)
            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))

            # create label for training (only do it one time)
            if not train_response_flag:
                # change control flag
                train_response_flag = True
                # get shape of output (i.e., response map)
                response_size = output.shape[2:4]
                # generate label and weight
                train_eltwise_label, train_instance_weight = create_label(response_size, config, use_gpu)

            # clear the gradient
            optimizer.zero_grad()

            # loss
            loss = net.weight_loss(output, train_eltwise_label, train_instance_weight)

            # backward
            loss.backward()

            # update parameter
            optimizer.step()

            # if binary network - clip weights
            if type == 'BINARY':
                net.clip_weights(-1, 1)

            # collect training loss
            train_loss.append(loss.data)

        # ------------------------------ saving model ------------------------------
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # torch.save(net, model_save_path + "SiamFC_" + str(i + 1) + "_model.pth")
        torch.save({
            'state_dict': net.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': i + 1,
        }, model_save_path + "SiamFC_" + str(i + 1) + "_model.pth")

        # ------------------------------ validation ------------------------------
        # indicate validation
        net.eval()

        # used to collect validation loss
        val_loss = []

        for j, data in enumerate(tqdm(val_loader)):

            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))

            # create label for validation (only do it one time)
            if not valid_response_flag:
                valid_response_flag = True
                response_size = output.shape[2:4]
                valid_eltwise_label, valid_instance_weight = create_label(response_size, config, use_gpu)

            # loss
            loss = net.weight_loss(output, valid_eltwise_label, valid_instance_weight)

            # collect validation loss
            val_loss.append(loss.data)

        mean_train_loss = np.mean(np.array(train_loss, dtype=np.float32))
        mean_val_loss = np.mean(np.array(val_loss, dtype=np.float32))
        print("Epoch %d   training loss: %f, validation loss: %f" % (i + 1,
                                                                     mean_train_loss,
                                                                     mean_val_loss))
        writer.add_scalar("Loss/train", mean_train_loss, i + 1)
        writer.add_scalar("Loss/val", mean_val_loss, i + 1)


def save_config(config, path="./"):
    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(path)
    with open(os.path.join(path, "train_config.txt"), 'w') as json_file:
        json.dump(config.__dict__, json_file, indent=4)


if __name__ == "__main__":

    # data_dir = "/home/vision/dw/ILSVRC2015_curated/Data/VID/train"
    # train_imdb = "/home/vision/orig_dp/siamtrackopt/ILSVRC15-curation/imdb_video_train.json"
    # val_imdb = "/home/vision/orig_dp/siamtrackopt/ILSVRC15-curation/imdb_video_val.json"


    # # ## FINN_W2_A2
    # exp_nbr = "FINN_W2_A2" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # save_config(Config(), f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A2)
    # writer.close()
    # #

    # # ## FINN_W2_A2_X2
    # exp_nbr = "FINN_W2_A2_X2" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # save_config(Config(), f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A2_X2)
    # writer.close()
    # #


    # # ## FINN_W2_A2_X3
    # exp_nbr = "FINN_W2_A2_X3" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A2_X3 = Config()
    # config_FINN_W2_A2_X3.val_videos = 400
    # save_config(config_FINN_W2_A2_X3, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A2_X3, config=config_FINN_W2_A2_X3)
    # writer.close()
    #
    # # ## FINN_W2_A8_X4
    # exp_nbr = "FINN_W2_A8_X4" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A8_X4 = Config()
    # config_FINN_W2_A8_X4.num_pairs = 2000
    # config_FINN_W2_A8_X4.val_videos = 100
    # save_config(config_FINN_W2_A8_X4, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A8_X4,
    #       config=config_FINN_W2_A8_X4)
    # writer.close()
    #
    #
    # # ## FINN_W2_A8_X4
    # exp_nbr = "FINN_W2_A8_X4" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A8_X4 = Config()
    # config_FINN_W2_A8_X4.val_videos = 400
    # save_config(config_FINN_W2_A8_X4, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A8_X4, config=config_FINN_W2_A8_X4)
    # writer.close()
    # #
    #
    # # ## FINN_W2_A4_X5
    # exp_nbr = "FINN_W2_A4_X5" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A4_X5 = Config()
    # config_FINN_W2_A4_X5.val_videos = 400
    # save_config(config_FINN_W2_A4_X5, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A4_X5, config=config_FINN_W2_A4_X5)
    # writer.close()
    # #

    # # ## FINN_W2_A16_X6
    # exp_nbr = "FINN_W2_A16_X6" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A16_X6 = Config()
    # config_FINN_W2_A16_X6.val_videos = 400
    # save_config(config_FINN_W2_A16_X6, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A16_X6, config=config_FINN_W2_A16_X6)
    # writer.close()
    #

    # # ## FINN_W2_A2_X10
    # exp_nbr = "FINN_W2_A2_X10" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A2_X10 = Config()
    # config_FINN_W2_A2_X10.val_videos = 400
    # save_config(config_FINN_W2_A2_X10, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A2_X10,
    #       config=config_FINN_W2_A2_X10)
    # writer.close()
    #
    # # ## FINN_W2_A2_X11
    # exp_nbr = "FINN_W2_A2_X11" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A2_X11 = Config()
    # config_FINN_W2_A2_X11.val_videos = 400
    # save_config(config_FINN_W2_A2_X11, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A2_X11,
    #       config=config_FINN_W2_A2_X11)
    # writer.close()

    # # ## FINN_W1_A1_X12
    # exp_nbr = "FINN_W1_A1_X12" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W1_A1_X12 = Config()
    # config_FINN_W1_A1_X12.val_videos = 400
    # save_config(config_FINN_W1_A1_X12, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W1_A1_X12,
    #       config=config_FINN_W1_A1_X12)
    # writer.close()
    #
    # # ## FINN_W1_A1_X13
    # exp_nbr = "FINN_W1_A1_X13" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W1_A1_X13 = Config()
    # config_FINN_W1_A1_X13.val_videos = 400
    # save_config(config_FINN_W1_A1_X13, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W1_A1_X13,
    #       config=config_FINN_W1_A1_X13)
    # writer.close()

    # # ## FINN_W2_A4_14
    # exp_nbr = "FINN_W2_A4_X14" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A4_X14 = Config()
    # config_FINN_W2_A4_X14.val_videos = 400
    # save_config(config_FINN_W2_A4_X14, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A4_X14,
    #       config=config_FINN_W2_A4_X14)
    # writer.close()
    #
    #
    # # ## FINN_W2_A8_15
    # exp_nbr = "FINN_W2_A8_X15" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A8_X15 = Config()
    # config_FINN_W2_A8_X15.val_videos = 400
    # save_config(config_FINN_W2_A8_X15, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A8_X15,
    #       config=config_FINN_W2_A8_X15)
    # writer.close()


    # # ## FINN_W2_A16_X16
    # exp_nbr = "FINN_W2_A16_X16" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A16_X16 = Config()
    # config_FINN_W2_A16_X16.val_videos = 400
    # save_config(config_FINN_W2_A16_X16, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A16_X16,
    #       config=config_FINN_W2_A16_X16)
    # writer.close()

    # # ## FINN_W2_A1_X17
    # exp_nbr = "FINN_W2_A1_X17" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A1_X17 = Config()
    # config_FINN_W2_A1_X17.val_videos = 400
    # save_config(config_FINN_W2_A1_X17, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A1_X17,
    #       config=config_FINN_W2_A1_X17)
    # writer.close()
    #
    # # ## FINN_W2_A1_X18
    # exp_nbr = "FINN_W2_A1_X18" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A1_X18 = Config()
    # config_FINN_W2_A1_X18.val_videos = 400
    # save_config(config_FINN_W2_A1_X18, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A1_X18,
    #       config=config_FINN_W2_A1_X18)
    # writer.close()

    # # ## FINN_W2_A2_X19
    # exp_nbr = "FINN_W2_A2_X19" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A2_X19 = Config()
    # config_FINN_W2_A2_X19.val_videos = 400
    # save_config(config_FINN_W2_A2_X19, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A2_X19,
    #       config=config_FINN_W2_A2_X19)
    # writer.close()

    # # ## FINN_W2_A4_X20
    # exp_nbr = "FINN_W2_A4_X20" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A4_X20 = Config()
    # config_FINN_W2_A4_X20.val_videos = 400
    # save_config(config_FINN_W2_A4_X20, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A4_X20,
    #       config=config_FINN_W2_A4_X20)
    # writer.close()
    #
    # # ## FINN_W2_A8_X21
    # exp_nbr = "FINN_W2_A8_X21" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A8_X21 = Config()
    # config_FINN_W2_A8_X21.val_videos = 400
    # save_config(config_FINN_W2_A8_X21, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A8_X21,
    #       config=config_FINN_W2_A8_X21)
    # writer.close()
    #
    # # ## FINN_W2_A16_X22
    # exp_nbr = "FINN_W2_A16_X22" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A16_X22 = Config()
    # config_FINN_W2_A16_X22.val_videos = 400
    # save_config(config_FINN_W2_A16_X22, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A16_X22,
    #       config=config_FINN_W2_A16_X22)
    # writer.close()

    # # ## FINN_W2_A1_X23
    # exp_nbr = "FINN_W2_A1_X23" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A1_X23 = Config()
    # config_FINN_W2_A1_X23.val_videos = 400
    # save_config(config_FINN_W2_A1_X23, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W2_A1_X23,
    #       config=config_FINN_W2_A1_X23)
    # writer.close()


    # # ## FINN_W32_A32_X24_FP
    # exp_nbr = "FINN_W32_A32_X24_FP" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W32_A32_X24_FP = Config()
    # config_FINN_W32_A32_X24_FP.val_videos = 400
    # save_config(config_FINN_W32_A32_X24_FP, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W32_A32_X24_FP,
    #       config=config_FINN_W32_A32_X24_FP)
    # writer.close()

    data_dir = "/home/vision/ILSVRC2015/ILSVRC2015_curated_z256_x130/Data/VID/train"
    train_imdb = "/home/vision/siamtrackopt/ILSVRC15-curation/imdb_video_train_z256_x130.json"
    val_imdb = "/home/vision/siamtrackopt/ILSVRC15-curation/imdb_video_val_z256_x130.json"

    # exp_nbr = "FINN_W32_A32_X25_FP_z256_x130" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W32_A32_X25_FP_z256_x130 = Config()
    # config_FINN_W32_A32_X25_FP_z256_x130.val_videos = 400
    # config_FINN_W32_A32_X25_FP_z256_x130.examplar_size = 130
    # config_FINN_W32_A32_X25_FP_z256_x130.instance_size = 256
    # config_FINN_W32_A32_X25_FP_z256_x130.score_size = 30
    # save_config(config_FINN_W32_A32_X25_FP_z256_x130, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/", type=NetType.FINN_W32_A32_X25_FP_z256_x130,
    #       config=config_FINN_W32_A32_X25_FP_z256_x130)
    # writer.close()

    # exp_nbr = "FINN_W2_A2_X26_z256_x130" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A2_X26_z256_x130 = Config()
    # config_FINN_W2_A2_X26_z256_x130.val_videos = 400
    # config_FINN_W2_A2_X26_z256_x130.examplar_size = 130
    # config_FINN_W2_A2_X26_z256_x130.instance_size = 256
    # config_FINN_W2_A2_X26_z256_x130.score_size = 30
    # save_config(config_FINN_W2_A2_X26_z256_x130, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/",
    #       type=NetType.FINN_W2_A2_X26_z256_x130,
    #       config=config_FINN_W2_A2_X26_z256_x130)
    # writer.close()
    #
    # exp_nbr = "FINN_W2_A8_X27_z256_x130" + "_" + str(dt.now())
    # writer = SummaryWriter(f"runs/{exp_nbr}")
    # config_FINN_W2_A8_X27_z256_x130 = Config()
    # config_FINN_W2_A8_X27_z256_x130.val_videos = 400
    # config_FINN_W2_A8_X27_z256_x130.examplar_size = 130
    # config_FINN_W2_A8_X27_z256_x130.instance_size = 256
    # config_FINN_W2_A8_X27_z256_x130.score_size = 30
    # save_config(config_FINN_W2_A8_X27_z256_x130, f"./models_{exp_nbr}/")
    # # training SiamFC network, using GPU by default
    # train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/",
    #       type=NetType.FINN_W2_A8_X27_z256_x130,
    #       config=config_FINN_W2_A8_X27_z256_x130)
    # writer.close()

    exp_nbr = "FINN_W2_A2_X26_z256_x130" + "_" + str(dt.now())
    writer = SummaryWriter(f"runs/{exp_nbr}")
    config_FINN_W2_A2_X26_z256_x130 = Config()
    config_FINN_W2_A2_X26_z256_x130.val_videos = 400
    config_FINN_W2_A2_X26_z256_x130.examplar_size = 130
    config_FINN_W2_A2_X26_z256_x130.instance_size = 256
    config_FINN_W2_A2_X26_z256_x130.score_size = 30
    save_config(config_FINN_W2_A2_X26_z256_x130, f"./models_{exp_nbr}/")
    # training SiamFC network, using GPU by default
    train(data_dir, train_imdb, val_imdb, model_save_path=f"./models_{exp_nbr}/",
          type=NetType.FINN_W2_A2_X26_z256_x130,
          config=config_FINN_W2_A2_X26_z256_x130)
    writer.close()




