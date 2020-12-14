import os
from math import sqrt
from Train.SiamNet import NetType


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_overlap(gt, bbx):
    """
    Calculate the Intersection are of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert gt['x1'] <= gt['x2']
    assert gt['y1'] <= gt['y2']
    assert bbx['x1'] <= bbx['x2']
    assert bbx['y1'] <= bbx['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(gt['x1'], bbx['x1'])
    y_top = max(gt['y1'], bbx['y1'])
    x_right = min(gt['x2'], bbx['x2'])
    y_bottom = min(gt['y2'], bbx['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    return intersection_area


def evaluate(model):
    groundtruth_parent_dir = '/home/vision/cfnet-validation'
    tracking_result_dir = os.path.join('/home/vision/orig_dp/siamtrackopt/Tracking/tracking_result/', str(model))
    tracking_result_dir_finn = os.path.join('/home/vision/orig_dp/siamtrackopt/FINN/tracking_result/', str(model))

    groundtruth_files = []
    tracking_result_files = []

    for r, d, f in os.walk(groundtruth_parent_dir):
        for folder in d:
            groundtruth_files.append(os.path.join(r, folder, 'groundtruth.txt'))

    for r, d, f in os.walk(tracking_result_dir):
        for file in f:
            if '.txt' in file:
                tracking_result_files.append(os.path.join(r, file))

    for r, d, f in os.walk(tracking_result_dir_finn):
        for file in f:
            if '.txt' in file:
                tracking_result_files.append(os.path.join(r, file))

    sequences_pairs = {}

    for gt in groundtruth_files:
        seq_name = gt.split('/')[-2]

        for result in tracking_result_files:
            if seq_name.lower() in result.lower():
                sequences_pairs[seq_name] = [gt, result]
                break

    evaluation = {}

    for k, v in sequences_pairs.items():
        gt_dir, res_dir = v
        gt_list = []
        res_list = []

        with open(gt_dir) as gt_file:
            for line in gt_file:
                gt_list.append(line)

        with open(res_dir) as res_file:
            for line in res_file:
                res_list.append(line)

        seq_IOU = []
        seq_center_error = []

        precision_threshold = 20.0
        center_error_within_threshold = 0

        for it in range(len(res_list)):
            res_bbox = [float(x) for x in res_list[it].split(' ')]
            gt_bbox = [float(x) for x in gt_list[it].split(',')]
            if len(gt_bbox) > 4:
                xs = [float(gt_bbox[0]), float(gt_bbox[2]), float(gt_bbox[4]), float(gt_bbox[6])]
                ys = [float(gt_bbox[1]), float(gt_bbox[3]), float(gt_bbox[5]), float(gt_bbox[7])]

                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)

                gt_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                # gt_bbox = [x_min, y_min, x_min+x_max, y_min+y_max]

            gt_xc = float(gt_bbox[0]) + float(gt_bbox[2])/2
            gt_yc = float(gt_bbox[1]) + float(gt_bbox[3])/2
            res_xc = float(res_bbox[0]) + float(res_bbox[2]) / 2
            res_yc = float(res_bbox[1]) + float(res_bbox[3]) / 2

            # seq_center_error.append(sqrt(abs(gt_xc - res_xc) + abs(gt_yc - res_yc)))
            seq_center_error.append(sqrt((gt_xc - res_xc) ** 2 + (gt_yc - res_yc) ** 2))

            bbox_1 = {'x1': gt_bbox[0], 'x2': gt_bbox[0] + gt_bbox[2], 'y1': gt_bbox[1], 'y2': gt_bbox[1] + gt_bbox[3]}

            bbox_2 = {'x1': res_bbox[0], 'x2': res_bbox[0] + res_bbox[2], 'y1': res_bbox[1],
                      'y2': res_bbox[1] + res_bbox[3]}

            seq_IOU.append(get_iou(bbox_1, bbox_2))


            # if seq_center_error[-1] > 50:
            #     break

            # The tracking precision is calculated as the ratio of the number of frames with center error below the
            # predetermined threshold
            # if seq_center_error[-1] <= precision_threshold:
            #     center_error_within_threshold += 1

        center_error = sum(seq_center_error) / len(seq_center_error)
        IOU = sum(seq_IOU) / len(seq_IOU)
        precision = sum([1 for i in range(len(seq_center_error)) if seq_center_error[i] < precision_threshold]) / len(seq_center_error)
        evaluation[k] = {'Center error': center_error, 'IOU': IOU, 'Precision': precision}
        print(k, ': ', 'Center error: ', center_error, '. IOU: ', IOU, 'Precision', precision)

    # Total: center error, IOU, precision
    total_center_error = 0.0
    total_IOU = 0.0
    total_precision = 0.0
    for k, v in evaluation.items():
        total_center_error += v['Center error']
        total_IOU += v['IOU']
        total_precision += v['Precision']

    total_center_error /= len(evaluation)
    total_IOU /= len(evaluation)
    total_precision /= len(evaluation)

    print('TOTAL NET STATISTICS')
    print('Mean center error: ', total_center_error, ' Mean IOU: ', total_IOU, ' Mean precision: ', total_precision)
    eval_dir = "./eval_result/"
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)
    with open(os.path.join(eval_dir, str(model) + ".txt"), "w") as file:
        file.writelines("".join('Mean center error: ' + str(total_center_error) +
                                ' Mean IOU: ' + str(total_IOU) + ' Mean precision: ' + str(total_precision)))

    with open(os.path.join(eval_dir, "comparison.txt"), "a") as file:
        file.writelines("".join(str(model) + ' Mean center error: ' + str(round(total_center_error, 3)) +
                                ' Mean IOU: ' + str(round(total_IOU, 3)) + ' Mean precision: ' + str(round(total_precision, 3))) + '\n')


def clear_comparison_file():
    eval_dir = "./eval_result/"
    with open(os.path.join(eval_dir, "comparison.txt"), "w") as file:
        file.close()


if __name__ == "__main__":
    clear_comparison_file()
    # evaluate(NetType.X1_1)
    # evaluate(NetType.X1_2)
    # evaluate(NetType.X1_3)
    # evaluate(NetType.X1_4)
    # evaluate(NetType.X1_5)
    # evaluate(NetType.X1_6)
    #
    # evaluate(NetType.X2_1)
    # evaluate(NetType.X2_2)
    # evaluate(NetType.X2_3)
    # evaluate(NetType.X2_4)
    # evaluate(NetType.X2_5)
    # evaluate(NetType.X2_6)
    #
    # evaluate(NetType.X3_1)
    # evaluate(NetType.X3_2)
    # evaluate(NetType.X3_3)
    # evaluate(NetType.X3_4)
    # evaluate(NetType.X3_5)
    # evaluate(NetType.X3_6)
    #
    # evaluate(NetType.X4_1)
    # evaluate(NetType.X4_2)
    # evaluate(NetType.X4_3)
    # evaluate(NetType.X4_4)
    # evaluate(NetType.X4_5)
    # evaluate(NetType.X4_6)
    #
    from FINN.SiamNet_finn import NetType
    evaluate(NetType.FINN_W2_A2_X3)
    evaluate(NetType.FINN_W2_A8_X4)
    evaluate(NetType.FINN_W2_A4_X5)
    evaluate(NetType.FINN_W2_A16_X6)
    #
    evaluate(NetType.FINN_W2_A2_X10)
    evaluate(NetType.FINN_W2_A2_X11)
    evaluate(NetType.FINN_W2_A2_X11_orig)
    evaluate(NetType.FINN_W1_A1_X12)
    evaluate(NetType.FINN_W1_A1_X13)
    #
    evaluate(NetType.FINN_W2_A4_X14)
    evaluate(NetType.FINN_W2_A8_X15)
    evaluate(NetType.FINN_W2_A16_X16)

    evaluate(NetType.FINN_W2_A1_X17)
    evaluate(NetType.FINN_W2_A1_X18)
    evaluate(NetType.FINN_W2_A2_X19)

    evaluate(NetType.FINN_W2_A4_X20)
    evaluate(NetType.FINN_W2_A8_X21)
    evaluate(NetType.FINN_W2_A16_X22)
    evaluate(NetType.FINN_W2_A1_X23)

    evaluate(NetType.FINN_W32_A32_X25_FP_z256_x130)
    evaluate(NetType.FINN_W32_A32_X24_FP)

    evaluate(NetType.FINN_W2_A2_X26_z256_x130)
    evaluate(NetType.FINN_W2_A8_X27_z256_x130)

    from FINN.final_SiamNet_finn import NetType
    evaluate(NetType.FINN_W2_A2_X28)
    evaluate(NetType.FINN_W2_A2_X29)
    evaluate(NetType.FINN_W2_A2_X30)
    evaluate(NetType.FINN_W2_A2_X31)








