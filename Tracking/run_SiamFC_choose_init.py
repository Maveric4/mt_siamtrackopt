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
from Train.SiamNet import SiamNet, NetType
from Train.Config import *
from tqdm import tqdm
from Tracking.Tracking_Utils import Recorder, Stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global var
import copy
X1, Y1, INIT_IMG = None, None, None
X2, Y2 = None, None
was_first_corner = False
INIT_MODE = True


def draw_circle(event, x, y, flags, param):
    global X1,Y1, INIT_IMG, was_first_corner, X2, Y2, INIT_MODE
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(INIT_IMG,(x,y), 2,(255,0,0),-1)
        cv2.imshow("Choose object to track", INIT_IMG)
        if not was_first_corner:
            X1, Y1 = x,y
            was_first_corner = True
        else:
            X2, Y2 = x, y
            INIT_MODE = False


# entry to evaluation of SiamFC
def run_tracker(p, net_type, net_base_path, model_name):
    global INIT_IMG, X1, X2, Y1, Y2
    """
    run tracker, return bounding result and speed
    """
    # load model # IF QUANTISED
    net = SiamNet(net_type)
    net.load_state_dict(torch.load(
        os.path.join("/home/vision/orig_dp/siamtrackopt/Train", net_base_path, model_name))['state_dict'])
    net = net.to(device)
    # evaluation mode
    net.eval()

    # load sequence
    img_list, target_position, target_size, gt_list = load_sequence(p.seq_base_path, p.video)


    # first frame - choose starting bbox
    img_uint8 = cv2.imread(img_list[0])
    INIT_IMG = copy.copy(img_uint8)

    img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    img_double = np.double(img_uint8)    # uint8 to float

    cv2.namedWindow('Choose object to track')
    cv2.setMouseCallback('Choose object to track', draw_circle)
    while INIT_MODE:
        cv2.imshow("Choose object to track", INIT_IMG)
        cv2.waitKey(1)
    cv2.destroyWindow("Choose object to track")
    target_position = np.array([(Y1 + Y2) / 2, (X1 + X2) / 2], dtype=np.double)
    target_size = np.array([Y2 - Y1, X2 - X1], dtype = np.double)


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

    recorder.close()
    print(stats.meanIoU())
    print(stats.meanCenterError())
    print(stats.meanPrecision())
    end_time = datetime.datetime.now()
    fps = len(img_list)/max(1.0, (end_time-start_time).seconds)

    return bboxes, fps



def get_specific_video_results(net_type, net_base_path, model_name, video_name, vis=False, save_to_file=False):
    # get the default parameters
    p = Config()
    p.visualization = vis
    p.save_to_file = save_to_file
    p.save_to_video = False

    p.video = video_name
    print("Processing %s ... " % p.video)
    bbox_result, fps = run_tracker(p, net_type, net_base_path, model_name)
    # fps for this video
    print("FPS: %d " % fps)


if __name__ == "__main__":
    
    #get_specific_video_results(NetType.X4_6, "models_X4.6_2020-10-04 10:59:27.799692", "SiamFC_30_model.pth", "tc_Cup_ce", vis=True, save_to_file=False)



