"""
Tool functiond for tracking evaluation
Written by Heng Fan
"""

import os
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import cv2
import glob
from math import sqrt
from Evaluation.evaluation import get_iou


class Stats:
    def __init__(self):
        self.IoU = []
        self.pr = []
        self.CE = []

    def meanIoU(self):
        return sum(self.IoU) / len(self.IoU)

    def meanPrecision(self):
        return sum(self.pr) / len(self.pr)

    def meanCenterError(self):
        return sum(self.CE) / len(self.CE)


class Recorder:
    def __init__(self, net_type, video_name):
        self.is_initialized = False
        self.net_type = net_type
        self.video_name = video_name
        self.recorder = None

    def set_options(self, img):
        height, width, layers = img.shape
        size = (width, height)
        if not os.path.isdir('/home/vision/orig_dp/siamtrackopt/Tracking/videos/' + self.net_type):
            os.mkdir('/home/vision/orig_dp/siamtrackopt/Tracking/videos/' + self.net_type)
        self.recorder = cv2.VideoWriter(
            '/home/vision/orig_dp/siamtrackopt/Tracking/videos/' + self.net_type + "/" + self.video_name + '.avi',
            cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        self.is_initialized = True

    def write(self, img):

        self.recorder.write(img)

    def close(self):
        self.recorder.release()


def cat_img(image_cat1, image_cat2, image_cat3):
    """
    concatenate three 1-channel images to one 3-channel image
    """
    image = np.zeros(shape = (image_cat1.shape[0], image_cat1.shape[1], 3), dtype=np.double)
    image[:, :, 0] = image_cat1
    image[:, :, 1] = image_cat2
    image[:, :, 2] = image_cat3

    return image


def get_bbox_list(vid_path):
    gt_path = os.path.join(vid_path, "groundtruth.txt")
    gt_list = []
    with open(gt_path) as gt_file:
        for line in gt_file:
            gt_list.append([float(x) for x in line.split(',')])
    return gt_list


def load_sequence(seq_root_path, seq_name):
    """
    load sequences;
    sequences should be in OTB format, or you can custom this function by yourself
    """
    img_dir = os.path.join(seq_root_path, seq_name)
    # img_dir = os.path.join(seq_root_path, seq_name, 'img/')
    gt_path = os.path.join(seq_root_path, seq_name, 'groundtruth.txt')
    # gt_path = os.path.join(seq_root_path, seq_name, 'groundtruth_rect.txt')

    img_list = glob.glob(img_dir + '/' + "*.jpg")
    img_list.sort()
    img_list = [os.path.join(img_dir, x) for x in img_list]

    gt = np.loadtxt(gt_path, delimiter=',')
    init_bbox = gt[0]
    if len(init_bbox) > 4:
        # print(init_bbox)
        # input()
        xs = [float(init_bbox[0]), float(init_bbox[2]), float(init_bbox[4]), float(init_bbox[6])]
        ys = [float(init_bbox[1]), float(init_bbox[3]), float(init_bbox[5]), float(init_bbox[7])]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        init_bbox = [x_min, y_min, x_max - x_min , y_max - y_min]

    if seq_name == "Tiger1":
        init_bbox = gt[5]

    init_x = init_bbox[0]
    init_y = init_bbox[1]
    init_w = init_bbox[2]
    init_h = init_bbox[3]

    target_position = np.array([init_y + init_h/2, init_x + init_w/2], dtype = np.double)
    target_sz = np.array([init_h, init_w], dtype = np.double)

    if seq_name == "David":
        img_list = img_list[299:]
    if seq_name == "Tiger1":
        img_list = img_list[5:]
    if seq_name == "Football1":
        img_list = img_list[0:74]

    gt_list = []
    with open(gt_path) as gt_file:
        for line in gt_file:
            gt = [float(x) for x in line.split(',')]
            if len(gt) > 4:
                xs = [float(gt[0]), float(gt[2]), float(gt[4]), float(gt[6])]
                ys = [float(gt[1]), float(gt[3]), float(gt[5]), float(gt[7])]

                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)

                gt = [x_min, y_min, x_max - x_min, y_max - y_min]
            gt_list.append(gt)
    return img_list, target_position, target_sz, gt_list


def visualize_tracking_result(img, res_bbox, fig_n, it=0, gt_bbox=None, save_info=None, stats=None, recorder=None):
    """
    visualize tracking result
    """
    gt_xc = float(gt_bbox[0]) + float(gt_bbox[2]) / 2
    gt_yc = float(gt_bbox[1]) + float(gt_bbox[3]) / 2
    res_xc = float(res_bbox[0]) + float(res_bbox[2]) / 2
    res_yc = float(res_bbox[1]) + float(res_bbox[3]) / 2

    center_error = sqrt((gt_xc - res_xc) ** 2 + (gt_yc - res_yc) ** 2)
    center_error_threshold = 20
    stats.CE.append(center_error)
    stats.pr.append(1 if center_error < center_error_threshold else 0)

    bbox_1 = {'x1': gt_bbox[0], 'x2': gt_bbox[0] + gt_bbox[2], 'y1': gt_bbox[1], 'y2': gt_bbox[1] + gt_bbox[3]}
    bbox_2 = {'x1': res_bbox[0], 'x2': res_bbox[0] + res_bbox[2], 'y1': res_bbox[1], 'y2': res_bbox[1] + res_bbox[3]}

    stats.IoU.append(get_iou(bbox_1, bbox_2))

    # print(gt_bbox)
    # print(res_bbox)
    # print(get_iou(bbox_1, bbox_2))
    # input()
    cv2.putText(img, 'mean IoU ' + str(round(stats.meanIoU(), 3)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, 'mean pr ' + str(round(stats.meanPrecision(), 3)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, 'mean CE ' + str(round(stats.meanCenterError(), 3)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    img_with_bbox = cv2.rectangle(img,
                                  (int(res_bbox[0]), int(res_bbox[1])),
                                  (int(res_bbox[0]+res_bbox[2]),
                                   int(res_bbox[1]+res_bbox[3])),
                                  color=(255, 0, 0),
                                  thickness=2)
    img_with_gt_bbox = cv2.rectangle(cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR),
                                     (int(gt_bbox[0]), int(gt_bbox[1])),
                                     (int(gt_bbox[0]+gt_bbox[2]),
                                      int(gt_bbox[1]+gt_bbox[3])),
                                     color=(0, 255, 0),
                                     thickness=2)
    cv2.imshow("Tracking result", img_with_gt_bbox)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()

    if save_info["save_to_file"]:
        if not os.path.isdir('/home/vision/orig_dp/siamtrackopt/Tracking/figs/'+ save_info["net_type"] + "/"):
            os.mkdir('/home/vision/orig_dp/siamtrackopt/Tracking/figs/'+ save_info["net_type"] + "/")
        if not os.path.isdir('/home/vision/orig_dp/siamtrackopt/Tracking/figs/'+ save_info["net_type"] + "/" + save_info["video_name"]):
            os.mkdir('/home/vision/orig_dp/siamtrackopt/Tracking/figs/'+ save_info["net_type"] + "/" + save_info["video_name"])
        cv2.imwrite('/home/vision/orig_dp/siamtrackopt/Tracking/figs/'+ save_info["net_type"] + "/" + save_info["video_name"] + "/{:04d}".format(it) + '.jpg', img_with_gt_bbox)

    if save_info["save_to_video"]:
        # recorder.write(cv2.imread('/home/vision/orig_dp/siamtrackopt/Tracking/figs/'+ save_info["net_type"] + "/" + save_info["video_name"] + "/{:04d}".format(it) + '.jpg'))
        recorder.write(img_with_gt_bbox)

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    """
    extract image crop
    """
    if original_sz is None:
        original_sz = model_sz

    sz = original_sz
    im_sz = im.shape
    # make sure the size is not too small
    assert (im_sz[0] > 2) & (im_sz[1] > 2), "The size of image is too small!"
    c = (sz+1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c)       # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[0] - c)       # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = max(0, 1-context_xmin)       # in python, index starts from 0
    top_pad = max(0, 1-context_ymin)
    right_pad = max(0, context_xmax - im_sz[1])
    bottom_pad = max(0, context_ymax - im_sz[0])

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    im_R = im[:, :, 0]
    im_G = im[:, :, 1]
    im_B = im[:, :, 2]

    # padding
    if (top_pad !=0) | (bottom_pad !=0) | (left_pad !=0) | (right_pad !=0):
        im_R = np.pad(im_R, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant', constant_values = avg_chans[0])
        im_G = np.pad(im_G, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant', constant_values = avg_chans[1])
        im_B = np.pad(im_B, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant', constant_values = avg_chans[2])

        im = cat_img(im_R, im_G, im_B)

    im_patch_original = im[int(context_ymin)-1:int(context_ymax), int(context_xmin)-1:int(context_xmax), :]

    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch_original, (int(model_sz), int(model_sz)), interpolation = cv2.INTER_CUBIC)
    else:
        im_patch = im_patch_original

    return im_patch


def make_scale_pyramid(im, target_position, in_side_scaled, out_side, avg_chans, p):
    """
    extract multi-scale image crops
    """
    in_side_scaled = np.round(in_side_scaled)
    pyramid = np.zeros((out_side, out_side, 3, p.num_scale), dtype = np.double)
    max_target_side = in_side_scaled[in_side_scaled.size-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side
    # size_in_search_area = beta * size_in_image
    # e.g. out_side = beta * min_target_side
    search_side = round(beta * max_target_side)

    search_region = get_subwindow_tracking(im, target_position, search_side, max_target_side, avg_chans)

    assert (round(beta * min_target_side) == out_side), "Error!"

    for s in range(p.num_scale):
        target_side = round(beta * in_side_scaled[s])
        search_target_position = np.array([1 + search_side/2, 1 + search_side/2], dtype = np.double)
        pyramid[:, :, :, s] = get_subwindow_tracking(search_region, search_target_position, out_side,
                                                   target_side, avg_chans)

    return pyramid


def tracker_eval(net, s_x, z_features, x_features, target_position, window, p):
    """
    do evaluation (i.e., a forward pass for search region)
    (This part is implemented as in the original Matlab version)
    """
    # compute scores search regions of different scales
    scores = net.xcorr(z_features, x_features)
    scores = scores.to("cpu")

    response_maps = scores.squeeze().permute(1, 2, 0).data.numpy()
    # for this one, the opencv resize function works fine
    response_maps_up = cv2.resize(response_maps, (response_maps.shape[0]*p.response_UP, response_maps.shape[0]*p.response_UP), interpolation=cv2.INTER_CUBIC)

    # choose the scale whose response map has the highest peak
    if p.num_scale > 1:
        current_scale_id =np.ceil(p.num_scale/2)
        best_scale = current_scale_id
        best_peak = float("-inf")
        for s in range(p.num_scale):
            this_response = response_maps_up[:, :, s]
            # penalize change of scale
            if s != current_scale_id:
                this_response = this_response * p.scale_penalty
            this_peak = np.max(this_response)
            if this_peak > best_peak:
                best_peak = this_peak
                best_scale = s
        response_map = response_maps_up[:, :, int(best_scale)]
    else:
        response_map = response_maps_up
        best_scale = 1
    # make the response map sum to 1
    response_map = response_map - np.min(response_map)
    if sum(sum(response_map)) != 0:
        response_map = response_map / sum(sum(response_map))

    # apply windowing
    response_map = (1 - p.w_influence) * response_map + p.w_influence * window
    p_corr = np.asarray(np.unravel_index(np.argmax(response_map), np.shape(response_map)))

    # avoid empty
    if p_corr[0] is None:
        p_corr[0] = np.ceil(p.score_size/2)
    if p_corr[1] is None:
        p_corr[1] = np.ceil(p.score_size/2)

    # Convert to crop-relative coordinates to frame coordinates
    # displacement from the center in instance final representation ...
    disp_instance_final = p_corr - np.ceil(p.score_size * p.response_UP / 2)
    # ... in instance input ...
    disp_instance_input = disp_instance_final * p.stride / p.response_UP
    # ... in instance original crop (in frame coordinates)
    disp_instance_frame = disp_instance_input * s_x / p.instance_size
    # position within frame in frame coordinates
    new_target_position = target_position + disp_instance_frame

    return new_target_position, best_scale