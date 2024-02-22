# import os
# from tqdm import tqdm
# from medpy import metric
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# pred = "/home/zhangzifan/MaintoCode/2023-4-17-01/lits/test/pred_unet/"
# GT = "/home/zhangzifan/MaintoCode/2023-4-17-01/lits/test/labels/"
#
# file_list = os.listdir(pred)
# pred_list = []
# gt_list = []
# for file in tqdm(file_list):
#     x = cv2.imread(os.path.join(pred, file))
#     y = cv2.imread(os.path.join(GT, file))
#     x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
#     y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
#     pred_list.append(x)
#     gt_list.append(y)
# predict = np.array(pred_list, dtype=float)
# target = np.array(gt_list, dtype=float)
#
# print(metric.binary.dc(predict, target))
#

import os
import numpy as np
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def dice_coefficient(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    return 2.0 * intersection.sum() / (pred_mask.sum() + gt_mask.sum())

# 加载预测图像和标签
pred_folder = "/home/zhangzifan/MaintoCode/2023-4-17-01/lits/test/pred_unet/"
pred_images = load_images_from_folder(pred_folder)

# 加载真实标签图像
gt_folder = "/home/zhangzifan/MaintoCode/2023-4-17-01/lits/test/labels/"
gt_images = load_images_from_folder(gt_folder)

# 创建一个字典来保存每个类别的Dice指标
dice_scores = {}
num_classes = 2
# 对于每个类别，计算Dice指标
for i in range(1, num_classes+1):
    pred_masks = [img == i for img in pred_images]
    gt_masks = [img == i for img in gt_images]
    dice_scores[i] = np.mean([dice_coefficient(pred_mask, gt_mask) for pred_mask, gt_mask in zip(pred_masks, gt_masks)])

# 打印每个类别的Dice指标
for i in range(1, num_classes+1):
    print(f"Class {i}: {dice_scores[i]}")

