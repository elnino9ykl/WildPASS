"""
metrics of fusing two datasets
"""
import numpy as np
#import cv2
from copy import deepcopy
import datetime
import logging
import os

def create_logger(dir):
    logger = logging.getLogger("Logger")
    log_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(dir, "eval_{}.log".format(log_time))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    hdlr.setLevel(logging.INFO)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


class FusedIoU():
    def __init__(self, ncls_s1=26, incl_id_s1=None,
                 remap_src=None, remap_dst=None, logger=None):
        """s1=MAP, s2=IDD"""
        self.ncls_s1 = ncls_s1
        self.ign_id_s1 = [x for x in range(ncls_s1) if x not in incl_id_s1]
        self.incl_id_s1 = incl_id_s1
        self.remap_src = remap_src
        self.remap_dst = remap_dst
        self.logger = logger

        self.reset()

    def reset(self):
        self.tp = np.zeros(len(self.incl_id_s1))
        self.fp = np.zeros(len(self.incl_id_s1))
        self.fn = np.zeros(len(self.incl_id_s1))

    def add_batch(self, x, y):
        """x: pred or fused_pred, y:gt, size is 1xHxW scatter to onehot"""
        x_onehot = (np.arange(self.ncls_s1) == x[0, ..., None]).astype(int).transpose(2, 0, 1) \
            if x.shape[0] == 1 else x
        y_onehot = (np.arange(self.ncls_s1) == y[0, ..., None]).astype(int).transpose(2, 0, 1) \
            if y.shape[0] == 1 else x

        ignores = 0
        if (self.ign_id_s1 is not None):
            ignores = np.take(y_onehot, self.ign_id_s1, axis=0)
            # --- from 18xHxW to 1xHxW as mask
            ignores = np.bitwise_or.reduce(ignores.astype(bool), axis=0)
            ignores = ignores[None, ...].astype(int)
            y_onehot = np.take(y_onehot, self.incl_id_s1, axis=0)
            x_onehot = np.take(x_onehot, self.incl_id_s1, axis=0)

        tp = (x_onehot * y_onehot).sum(axis=1).sum(axis=1)
        fp = (x_onehot * (1 - y_onehot - ignores)).sum(axis=1).sum(axis=1)
        fn = ((1 - x_onehot) * (y_onehot)).sum(axis=1).sum(axis=1)

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def fuse(self, gt, pred_s1, pred_s2, filename):
        """softmax as 1xnclsxHxW, then 1xHxW scatter to onehot nclsxHxW, gt is labelled as MAP"""
        # --- filename
        filename = filename[0].split("leftImg8bit/val/cs/")[1]

        pred_s1 = pred_s1[0]
        pred_s2 = pred_s2[0]

        # --- argmax
        argmax_s1 = np.argmax(pred_s1, axis=0)
        argmax_s2 = np.argmax(pred_s2, axis=0)
        # --- argsecondmax
        # argmax_sec_s1 = np.argsort(pred_s1, axis=0)[-2, ...]
        # argmax_sec_s2 = np.argsort(pred_s2, axis=0)[-2, ...]

        # --- maximum prob value
        amax_s1 = np.amax(pred_s1, axis=0)
        amax_s2 = np.amax(pred_s2, axis=0)
        # --- second maximun prob value
        amax_sec_s1 = np.sort(pred_s1, axis=0)[-2, ...]
        amax_sec_s2 = np.sort(pred_s2, axis=0)[-2, ...]

        # ==== fusion w.r.t confidence
        # --- confidence option 1: ratio(max, second_max) larger and equal
        ratio_s1 = amax_s1 / (amax_sec_s1 + 1e-5)
        ratio_s2 = amax_s2 / (amax_sec_s2 + 1e-5)
        larger_mask = ratio_s1 <= ratio_s2
        # --- confidence option 2: amax larger and equal
        # larger_mask = amax_s1 <= amax_s2
        # --- confidence option 3: var smaller, means smaller uncertainty
        # larger_mask = np.var(pred_s1, axis=0) >= np.var(pred_s2, axis=0)
        # --- TODO: confidence option 4: deep essembel

        replace_mask_s1 = np.zeros_like(larger_mask)
        new_pred_s1 = deepcopy(argmax_s1)

        remap_src = np.array(self.remap_src)
        for i, row in enumerate(larger_mask):
            for j, col in enumerate(row):
                # ---  if pixel of source_2 are in selected 15 classes
                if col and argmax_s2[i, j] in remap_src[..., 1]:
                    idx_remap_src = remap_src[..., 1].tolist().index(argmax_s2[i, j])
                    # --- if argmax_s1[i, j] != remap_src[..., 0][idx_remap_src]:
                    # --- and pred_MAP is not the "curb_MAP_13", "crosswalk_MAP_24" and "marking_MAP_23" [reserved]
                    if argmax_s1[i, j] not in (remap_src[..., 0][idx_remap_src], 13, 23, 24):
                        new_pred_s1[i, j] = remap_src[..., 0][idx_remap_src]
                        replace_mask_s1[i, j] = True

                # --- if pixel are (marking_MAP_23, road_IDD_0), then replace marking_MAP_23 with road_MAP_11
                if (argmax_s2[i, j] == 0) and (argmax_s1[i, j] == 23):
                    new_pred_s1[i, j] = 11
                    replace_mask_s1[i, j] = True

        # --- eval new pred of source 1 MAP
        self.add_batch(new_pred_s1[None, ...], gt)

        '''
        # --- plot uncertainty
        is_plot_unc = False
        if is_plot_unc:
            # --- uncertainty: var(prob)
            unc_s1 = np.var(pred_s1, axis=0)
            unc_s1 *= 255.0 / (unc_s1.max() + 1e-5)
            unc_s1 = 255 - unc_s1
            plot_path = "./plots/" + filename.replace('.png', '_unc_MAP.png')
            unc_s1_jet = cv2.applyColorMap(unc_s1.astype(np.uint8), colormap=cv2.COLORMAP_BONE)
            unc_s1_jet[replace_mask_s1] = (0, 255, 0)
            cv2.imwrite(plot_path, unc_s1_jet)
        '''
        return new_pred_s1
        '''
            # --- IDD
            # unc_s2 = np.var(pred_s2, axis=0)
            # unc_s2 *= 255.0 / (unc_s2.max() + 1e-5)
            # unc_s2 = 255 - unc_s2
            # plot_path = "./plots/" + filename.replace('.png', '_unc_IDD.png')
            # unc_s2_jet = cv2.applyColorMap(unc_s2.astype(np.uint8), colormap=cv2.COLORMAP_JET)
            # cv2.imwrite(plot_path, unc_s2_jet)
        '''

    def get_iou(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        acc = num / (self.tp + self.fn + 1e-15)
        return np.nanmean(iou), iou, np.nanmean(acc), acc
