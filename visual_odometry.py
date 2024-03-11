# -*- coding: utf-8 -*-
# @Time : 2023/3/7 16:52
# @Author : xiao cong
# @Description :

from pylab import *
import math
import cv2

def get_im_xSums(rawImg, ODO_IMG_YAW_ROT_Y_RANGE, ODO_IMG_YAW_ROT_X_RANGE):
    # gets the scanline intensity which is returned as im_xsums of the sub_image
    # specified by the parameters initialized in the constructor

    subRawImg = rawImg[ODO_IMG_YAW_ROT_Y_RANGE, ODO_IMG_YAW_ROT_X_RANGE]  # 截取部分图像
    img_xsums = asarray(sum(subRawImg, 0), dtype='float64')  # 行向量
    img_xsums = img_xsums / (sum(img_xsums) / len(img_xsums))  # 归一化 0~1
    return img_xsums


def compare_segments(seg1, seg2, shift_length, compare_length_of_intensity):
    # assume a large difference
    minimum_difference_intensity = 1e6

    # for each offset, sum the abs difference between the two segments
    for offset in range(0, shift_length + 1):

        compare_difference_segments = abs(seg1[offset: compare_length_of_intensity] -
                                          seg2[0: compare_length_of_intensity - offset])

        sum_compare_difference_segments = float(sum(compare_difference_segments)) / float(
            compare_length_of_intensity - offset)

        if sum_compare_difference_segments < minimum_difference_intensity:
            minimum_difference_intensity = sum_compare_difference_segments
            minimum_offset = offset


    for offset in range(1, shift_length + 1):

        compare_difference_segments = abs(
            seg1[0:compare_length_of_intensity - offset] - seg2[offset:compare_length_of_intensity])
        sum_compare_difference_segments = float(sum(compare_difference_segments)) / float(
            compare_length_of_intensity - offset)
        if sum_compare_difference_segments < minimum_difference_intensity:
            minimum_difference_intensity = sum_compare_difference_segments
            minimum_offset = -offset

    out_minimum_offset = minimum_offset
    out_minimum_difference_intensity = minimum_difference_intensity  # 输出最小偏差及其对应的offset

    #print(out_minimum_offset, out_minimum_difference_intensity)

    return out_minimum_offset, out_minimum_difference_intensity


class VisualOdometry:
    def __init__(self, **kwargs):

        self.PREV_TRANS_V_IMG_X_SUMS = kwargs.pop("PREV_TRANS_V_IMG_X_SUMS", np.zeros(104))
        self.PREV_YAW_ROT_V_IMG_X_SUMS = kwargs.pop("PREV_YAW_ROT_V_IMG_X_SUMS", np.zeros(130))
        self.PREV_HEIGHT_V_IMG_Y_SUMS = kwargs.pop("PREV_HEIGHT_V_IMG_Y_SUMS", np.zeros(60))
        self.ODO_IMG_YAW_ROT_Y_RANGE = kwargs.pop("ODO_IMG_YAW_ROT_Y_RANGE", slice(31, 91))
        self.ODO_IMG_YAW_ROT_X_RANGE = kwargs.pop("ODO_IMG_YAW_ROT_X_RANGE", slice(16, 146))
        self.ODO_SHIFT_MATCH_HORI = kwargs.pop("ODO_SHIFT_MATCH_HORI", 26)
        self.ODO_YAW_ROT_V_SCALE = kwargs.pop("ODO_YAW_ROT_V_SCALE", 1)
        self.ODO_TRANS_V_SCALE = kwargs.pop("ODO_TRANS_V_SCALE", 30)
        self.ODO_IMG_HEIGHT_V_Y_RANGE = kwargs.pop("ODO_IMG_HEIGHT_V_Y_RANGE", slice(11, 111))
        self.ODO_IMG_HEIGHT_V_X_RANGE = kwargs.pop("ODO_IMG_HEIGHT_V_X_RANGE", slice(11, 151))

        self.FOV_HORI_DEGREE = kwargs.pop("FOV_HORI_DEGREE", 81.5)

        self.MAX_YAW_ROT_V_THRESHOLD = kwargs.pop("MAX_YAW_ROT_V_THRESHOLD", 2.5)
        self.MAX_TRANS_V_THRESHOLD = kwargs.pop("MAX_TRANS_V_THRESHOLD", 0.5)
        self.PREV_YAW_ROT_V = kwargs.pop("PREV_YAW_ROT_V", 0)
        self.PREV_TRANS_V = kwargs.pop("PREV_TRANS_V", 0.025)
        self.PREV_HEIGHT_V = kwargs.pop("PREV_HEIGHT_V", 0)

        self.ODO_SHIFT_MATCH_VERT = kwargs.pop("ODO_SHIFT_MATCH_VERT", 20)
        self.ODO_HEIGHT_V_SCALE = kwargs.pop("ODO_HEIGHT_V_SCALE", 20)
        self.MAX_HEIGHT_V_THRESHOLD = kwargs.pop("MAX_HEIGHT_V_THRESHOLD", 0.45)
        self.DEGREE_TO_RADIAN = np.pi / 180        # 角度转弧度

        self.odo = [0.0, 0.0, 0.0, 0]
        self.delta = [0, 0, 0]  # [transV, yawrotV, heightV]

    def update(self, RawImg):
        """ Estimate the translation and rotation of the viewer, given a new
            image sample of the environment

            (Matlab version: equivalent to rs_visual_odometry)
        """


        # start to compute the horizontal rotational velocity (yaw)  # 偏航角

        subRawImage = RawImg[self.ODO_IMG_YAW_ROT_Y_RANGE, self.ODO_IMG_YAW_ROT_X_RANGE]
        horiDegPerPixel = self.FOV_HORI_DEGREE / subRawImage.shape[1]
        imageXSums = sum(subRawImage, 0)

        avgIntensity = float64(sum(imageXSums)) / len(imageXSums)
        imageXSums = imageXSums / avgIntensity          # 归一化

        [minOffsetYawRot, minDiffIntensityRot] = compare_segments(imageXSums,
                                                self.PREV_YAW_ROT_V_IMG_X_SUMS,
                                                self.ODO_SHIFT_MATCH_HORI,
                                                len(imageXSums))
        #print(minOffsetYawRot)

        yawRotV = self.ODO_YAW_ROT_V_SCALE * minOffsetYawRot * horiDegPerPixel * self.DEGREE_TO_RADIAN     # 偏航角

        if abs(yawRotV) > self.MAX_YAW_ROT_V_THRESHOLD:      # 若值太大，就取上一帧的角速度作为当前角速度
            yawRotV = self.PREV_YAW_ROT_V
        else:
            self.PREV_YAW_ROT_V = yawRotV

        self.PREV_YAW_ROT_V_IMG_X_SUMS = imageXSums
        self.PREV_TRANS_V_IMG_X_SUMS = imageXSums

        # start to compute total translational velocity
        transV = minDiffIntensityRot * self.ODO_TRANS_V_SCALE

        if transV > self.MAX_TRANS_V_THRESHOLD:
            transV = self.PREV_TRANS_V
        else:
            self.PREV_TRANS_V = transV

        # start to compute the height change velocity
        subRawImg = RawImg[self.ODO_IMG_HEIGHT_V_Y_RANGE, self.ODO_IMG_HEIGHT_V_X_RANGE]

        if minOffsetYawRot > 0:
            subRawImg = subRawImg[:, minOffsetYawRot + 1:]
        else:
            subRawImg = subRawImg[:, 1: - (-minOffsetYawRot)]

        imageYSums = sum(subRawImage, 1)       #  按行求和
        avgIntensity = float64(sum(imageYSums)) / len(imageXSums)
        imageYSums = imageYSums / avgIntensity

        [minOffsetHeightV, minDiffIntensityHeight] = compare_segments(imageYSums,
                                                                  self.PREV_HEIGHT_V_IMG_Y_SUMS,
                                                                  self.ODO_SHIFT_MATCH_VERT,
                                                                  len(imageYSums))

        if minOffsetHeightV < 0:
            minDiffIntensityHeight = - minDiffIntensityHeight

        if minOffsetHeightV > 3:
            heightV = self.ODO_HEIGHT_V_SCALE * minDiffIntensityHeight
        else:
            heightV = 0

        if abs(heightV) > self.MAX_HEIGHT_V_THRESHOLD:
            heightV = self.PREV_HEIGHT_V
        else:
            self.PREV_HEIGHT_V = heightV

        self.PREV_HEIGHT_V_IMG_Y_SUMS = imageYSums

        self.odo[3] += yawRotV
        self.odo[0] += transV * cos(self.odo[3])  # xcoord
        self.odo[1] += transV * sin(self.odo[3])  # ycoord
        self.odo[2] += heightV                    # zcoord

        self.delta = [transV, yawRotV, heightV]
        # print(self.odo)

        return self.odo