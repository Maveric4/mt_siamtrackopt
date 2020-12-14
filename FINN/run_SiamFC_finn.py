#from Tracking.Config import *
from Tracking.Tracking_Utils import *
# from Tracking.SiamNet import *
import os
import numpy as np
import torchvision.transforms.functional as F
import cv2
import datetime
import torch
from torch.autograd import Variable
from FINN.SiamNet_finn import SiamNet, NetType
from Train.Config import *
from tqdm import tqdm
from Tracking.Tracking_Utils import Recorder, Stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# entry to evaluation of SiamFC
def run_tracker(p, net_type, net_base_path, model_name):
    """
    run tracker, return bounding result and speed
    """
    # load model # IF QUANTISED
    net = SiamNet(net_type)
    net.load_state_dict(torch.load(
        os.path.join("/home/vision/orig_dp/siamtrackopt/FINN", net_base_path, model_name))['state_dict'])
    net = net.to(device)
    # evaluation mode
    net.eval()

    # load sequence
    img_list, target_position, target_size, gt_list = load_sequence(p.seq_base_path, p.video)

    # first frame
    img_uint8 = cv2.imread(img_list[0])
    img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    img_double = np.double(img_uint8)    # uint8 to float

    # compute avg for padding
    avg_chans = np.mean(img_double, axis=(0, 1))

    wc_z = target_size[1] + p.context_amount * sum(target_size)
    hc_z = target_size[0] + p.context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.examplar_size / s_z

    # crop examplar z in the first frame
    z_crop = get_subwindow_tracking(img_double, target_position, p.examplar_size, round(s_z), avg_chans)

    z_crop = np.uint8(z_crop)  # you need to convert it to uint8
    # convert image to tensor
    z_crop_tensor = 255.0 * F.to_tensor(z_crop).unsqueeze(0)

    d_search = (p.instance_size - p.examplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    # arbitrary scale saturation
    min_s_x = p.scale_min * s_x
    max_s_x = p.scale_max * s_x

    # generate cosine window
    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size * p.response_UP), np.hanning(p.score_size * p.response_UP))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size * p.response_UP, p.score_size * p.response_UP))
    window = window / sum(sum(window))

    # pyramid scale search
    scales = p.scale_step**np.linspace(-np.ceil(p.num_scale/2), np.ceil(p.num_scale/2), p.num_scale)

    # extract feature for examplar z
    z_features = net.conv_features(Variable(z_crop_tensor).to(device))
    z_features = z_features.repeat(p.num_scale, 1, 1, 1)

    # do tracking
    bboxes = np.zeros((len(img_list), 4), dtype=np.double)  # save tracking result
    stats = Stats()
    recorder = Recorder(str(net_type), p.video)
    start_time = datetime.datetime.now()
    for i in tqdm(range(0, len(img_list))):
        if i > 0:
            # do detection
            # currently, we only consider RGB images for tracking
            img_uint8 = cv2.imread(img_list[i])
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
            img_double = np.double(img_uint8)  # uint8 to float

            scaled_instance = s_x * scales
            scaled_target = np.zeros((2, scales.size), dtype = np.double)
            scaled_target[0, :] = target_size[0] * scales
            scaled_target[1, :] = target_size[1] * scales

            # extract scaled crops for search region x at previous target position
            x_crops = make_scale_pyramid(img_double, target_position, scaled_instance, p.instance_size, avg_chans, p)

            # get features of search regions
            x_crops_tensor = torch.FloatTensor(x_crops.shape[3], x_crops.shape[2], x_crops.shape[1], x_crops.shape[0])
            # response_map = SiameseNet.get_response_map(z_features, x_crops)
            for k in range(x_crops.shape[3]):
                tmp_x_crop = x_crops[:, :, :, k]
                tmp_x_crop = np.uint8(tmp_x_crop)
                # numpy array to tensor
                x_crops_tensor[k, :, :, :] = 255.0 * F.to_tensor(tmp_x_crop).unsqueeze(0)

            # get features of search regions
            x_features = net.conv_features(Variable(x_crops_tensor).to(device))

            # evaluate the offline-trained network for exemplar x features
            target_position, new_scale = tracker_eval(net, round(s_x), z_features, x_features, target_position, window, p)

            # scale damping and saturation
            s_x = max(min_s_x, min(max_s_x, (1 - p.scale_LR) * s_x + p.scale_LR * scaled_instance[int(new_scale)]))
            target_size = (1 - p.scale_LR) * target_size + p.scale_LR * np.array([scaled_target[0, int(new_scale)], scaled_target[1, int(new_scale)]])

        rect_position = np.array([target_position[1]-target_size[1]/2, target_position[0]-target_size[0]/2, target_size[1], target_size[0]])

        if p.visualization:
            if not recorder.is_initialized:
                recorder.set_options(img_uint8)
            save_info = {"net_type": str(net_type),
                         "video_name": p.video,
                         "save_to_file": p.save_to_file,
                         "save_to_video": p.save_to_video}
            visualize_tracking_result(img_uint8, rect_position, 1, i, gt_list[i], save_info, stats, recorder)

        # output bbox in the original frame coordinates
        o_target_position = target_position
        o_target_size = target_size
        bboxes[i,:] = np.array([o_target_position[1]-o_target_size[1]/2, o_target_position[0]-o_target_size[0]/2, o_target_size[1], o_target_size[0]])

    if p.visualization:
        recorder.close()
        # print(stats.meanIoU())
        # print(stats.meanCenterError())
        # print(stats.meanPrecision())
    end_time = datetime.datetime.now()
    fps = len(img_list)/max(1.0, (end_time-start_time).seconds)

    return bboxes, fps


def get_results(net_type, net_base_path, model_name, vis=False, p=Config()):
    # get the default parameters
    # p = Config()
    # load all videos
    all_videos = os.listdir(p.seq_base_path)
    p.bbox_output = True
    p.bbox_output_path = os.path.join("./tracking_result/", str(net_type))
    p.visualization = vis

    if p.bbox_output:
        if not os.path.exists(p.bbox_output_path):
            os.makedirs(p.bbox_output_path)

    fps_all = .0

    for it, video in enumerate(all_videos):
        p.video = video
        print("Processing %s ... " % p.video)
        print("{} / {}".format(it, len(all_videos)))
        bbox_result, fps = run_tracker(p, net_type, net_base_path, model_name)
        # fps for this video
        print("FPS: %d " % fps)
        # saving tracking results
        if p.bbox_output:
            np.savetxt(os.path.join(p.bbox_output_path, p.video.lower() + '_SiamFC.txt'), bbox_result, fmt='%.3f')
        fps_all = fps_all + fps

    avg_fps = fps_all / len(all_videos)
    print("Average FPS: %f" % avg_fps)


def get_specific_video_results(net_type, net_base_path, model_name, video_name, vis=False, save_to_file=False):
    # get the default parameters
    p = Config()
    p.visualization = vis
    p.save_to_file = save_to_file
    p.save_to_video = True

    p.video = video_name
    print("Processing %s ... " % p.video)
    bbox_result, fps = run_tracker(p, net_type, net_base_path, model_name)
    # fps for this video
    print("FPS: %d " % fps)


if __name__ == "__main__":
    # get_results(NetType.FINN_W2_A2_X3, "models_FINN_W2_A2_X3_2020-10-10 01:13:09.567536", "SiamFC_30_model.pth")
    # #NetType.FINN_W2_A8_X4 Mean center error: 368.377 Mean IOU: 0.134 Mean precision: 0.187
    # get_results(NetType.FINN_W2_A8_X4, "models_FINN_W2_A8_X4_2020-10-10 02:40:01.940870", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W2_A8_X4, "models_FINN_W2_A8_X4_2020-10-10 03:22:51.837914", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W2_A4_X5, "models_FINN_W2_A4_X5_2020-10-10 04:49:38.890021", "SiamFC_30_model.pth")

    # get_results(NetType.FINN_W2_A16_X6, "models_FINN_W2_A16_X6_2020-10-10 13:25:50.071744", "SiamFC_16_model.pth")

    # get_results(NetType.FINN_W2_A2_X10, "models_FINN_W2_A2_X10_2020-10-12 15:01:10.568501", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W2_A2_X11, "models_FINN_W2_A2_X11_2020-10-12 16:25:54.011404", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W1_A1_X12, "models_FINN_W1_A1_X12_2020-10-12 19:26:19.580224", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W1_A1_X13, "models_FINN_W1_A1_X13_2020-10-12 20:45:47.227679", "SiamFC_30_model.pth")

    # get_results(NetType.FINN_W2_A4_X14, "models_FINN_W2_A4_X14_2020-10-13 21:58:53.572488", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W2_A8_X15, "models_FINN_W2_A8_X15_2020-10-13 23:24:52.019560", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W2_A16_X16, "models_FINN_W2_A16_X16_2020-10-14 00:53:54.411651", "SiamFC_30_model.pth")


    # get_results(NetType.FINN_W2_A2_X11, "zero_act_magic/", "converted_model.pth")
    # get_results(NetType.FINN_W2_A1_X17, "models_FINN_W2_A1_X17_2020-10-15 23:02:20.433908/", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W2_A1_X18, "models_FINN_W2_A1_X18_2020-10-16 00:26:11.136468/", "SiamFC_30_model.pth")

    # get_results(NetType.FINN_W2_A2_X19, "models_FINN_W2_A2_X19_2020-10-16 12:51:25.954844", "SiamFC_30_model.pth")

    # get_results(NetType.FINN_W2_A4_X20, "models_FINN_W2_A4_X20_2020-10-16 16:53:59.432416", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W2_A8_X21, "models_FINN_W2_A8_X21_2020-10-16 18:21:37.054209", "SiamFC_30_model.pth")
    # get_results(NetType.FINN_W2_A16_X22, "models_FINN_W2_A16_X22_2020-10-16 19:48:42.963340", "SiamFC_30_model.pth")

    # get_results(NetType.FINN_W2_A1_X23, "models_FINN_W2_A1_X23_2020-10-17 11:07:38.293934", "SiamFC_30_model.pth")



    # p = Config()
    # p.instance_size = 256
    # p.examplar_size = 130
    # p.score_size = 22
    # get_results(NetType.FINN_W32_A32_X25_FP_z256_x130, "models_FINN_W32_A32_X25_FP_z256_x130_2020-10-20 00:22:19.317682", "SiamFC_30_model.pth", p=p)
    #
    # get_results(NetType.FINN_W32_A32_X24_FP, "models_FINN_W32_A32_X24_FP_2020-10-19 23:00:19.555791", "SiamFC_30_model.pth")

    # p = Config()
    # p.instance_size = 256
    # p.examplar_size = 130
    # p.score_size = 22
    # get_results(NetType.FINN_W2_A2_X26_z256_x130, "models_FINN_W2_A2_X26_z256_x130_2020-10-20 17:15:50.176674",
    #             "SiamFC_30_model.pth", p=p)
    #
    # p = Config()
    # p.instance_size = 256
    # p.examplar_size = 130
    # p.score_size = 22
    # get_results(NetType.FINN_W2_A8_X27_z256_x130, "models_FINN_W2_A8_X27_z256_x130_2020-10-20 18:46:17.581915",
    #             "SiamFC_30_model.pth", p=p)



