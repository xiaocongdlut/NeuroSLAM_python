# -*- coding: utf-8 -*-
# @Time : 2023/3/1 16:20
# @Author : xiao cong
# @Description :

import numpy as np
import math

global VT

class yawHeightHdcNetwork:

    # ***********************************************************
    # 定义基本参数
    def __init__(self, **kwargs):

        # The dimension of yaw in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_Y_DIM = kwargs.pop("YAW_HEIGHT_HDC_Y_DIM", 36)

        # The dimension of height in yaw_height_hdc network
        self.YAW_HEIGHT_HDC_H_DIM = kwargs.pop("YAW_HEIGHT_HDC_H_DIM", 36)

        # The dimension of local excitation weight matrix for yaw
        self.YAW_HEIGHT_HDC_EXCIT_Y_DIM = kwargs.pop("YAW_HEIGHT_HDC_EXCIT_Y_DIM", 8)

        # The dimension of local excitation weight matrix for height
        self.YAW_HEIGHT_HDC_EXCIT_H_DIM = kwargs.pop("YAW_HEIGHT_HDC_EXCIT_H_DIM", 8)

        # The dimension of local inhibition weight matrix for yaw
        self.YAW_HEIGHT_HDC_INHIB_Y_DIM = kwargs.pop("YAW_HEIGHT_HDC_INHIB_Y_DIM", 5)

        # The dimension of local inhibition weight matrix for height
        self.YAW_HEIGHT_HDC_INHIB_H_DIM = kwargs.pop("YAW_HEIGHT_HDC_INHIB_H_DIM", 5)

        # The global inhibition value
        self.YAW_HEIGHT_HDC_GLOBAL_INHIB = kwargs.pop("YAW_HEIGHT_HDC_GLOBAL_INHIB", 0.0002)

        # amount of energy injected when a view template is re-seen
        self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY = kwargs.pop("YAW_HEIGHT_HDC_VT_INJECT_ENERGY", 0.001)

        # Variance of Excitation and Inhibition in XY and THETA respectively
        self.YAW_HEIGHT_HDC_EXCIT_Y_VAR = kwargs.pop("YAW_HEIGHT_HDC_EXCIT_Y_VAR", 1.9)
        self.YAW_HEIGHT_HDC_EXCIT_H_VAR = kwargs.pop("YAW_HEIGHT_HDC_EXCIT_H_VAR", 1.9)
        self.YAW_HEIGHT_HDC_INHIB_Y_VAR = kwargs.pop("YAW_HEIGHT_HDC_INHIB_Y_VAR", 3.0)
        self.YAW_HEIGHT_HDC_INHIB_H_VAR = kwargs.pop("YAW_HEIGHT_HDC_INHIB_H_VAR", 3.0)

        # The scale of rotation velocity of yaw
        self.YAW_ROT_V_SCALE = kwargs.pop("YAW_ROT_V_SCALE", 1)

        # The scale of rotation velocity of height
        self.HEIGHT_V_SCALE = kwargs.pop("HEIGHT_V_SCALE", 1)

        # packet size for wrap, the left and right activity cells near
        self.YAW_HEIGHT_HDC_PACKET_SIZE = kwargs.pop("YAW_HEIGHT_HDC_PACKET_SIZE", 5)

        self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF = math.floor(self.YAW_HEIGHT_HDC_EXCIT_Y_DIM / 2)
        self.YAW_HEIGHT_HDC_EXCIT_H_DIM_HALF = math.floor(self.YAW_HEIGHT_HDC_EXCIT_H_DIM / 2)
        self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF = math.floor(self.YAW_HEIGHT_HDC_INHIB_Y_DIM / 2)
        self.YAW_HEIGHT_HDC_INHIB_H_DIM_HALF = math.floor(self.YAW_HEIGHT_HDC_INHIB_H_DIM / 2)

        # The yaw theta size of each unit in radian
        self.YAW_HEIGHT_HDC_Y_TH_SIZE = (2 * np.pi) / self.YAW_HEIGHT_HDC_Y_DIM

        # The yaw theta size of each unit in radian
        self.YAW_HEIGHT_HDC_H_SIZE = (2 * np.pi) / self.YAW_HEIGHT_HDC_H_DIM

        self.YAW_HEIGHT_HDC_Y_SUM_SIN_LOOKUP = np.sin(np.arange(1, self.YAW_HEIGHT_HDC_Y_DIM + 1) * self.YAW_HEIGHT_HDC_Y_TH_SIZE)
        self.YAW_HEIGHT_HDC_Y_SUM_COS_LOOKUP = np.cos(np.arange(1, self.YAW_HEIGHT_HDC_Y_DIM + 1) * self.YAW_HEIGHT_HDC_Y_TH_SIZE)

        self.YAW_HEIGHT_HDC_H_SUM_SIN_LOOKUP = np.sin(np.arange(1, self.YAW_HEIGHT_HDC_H_DIM + 1) * self.YAW_HEIGHT_HDC_H_SIZE)
        self.YAW_HEIGHT_HDC_H_SUM_COS_LOOKUP = np.cos(np.arange(1, self.YAW_HEIGHT_HDC_H_DIM + 1) * self.YAW_HEIGHT_HDC_H_SIZE)

        self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP = list(range(self.YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF, self.YAW_HEIGHT_HDC_Y_DIM)) + \
                               list(range(0, self.YAW_HEIGHT_HDC_Y_DIM)) + list(range(0, self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF))
        self.YAW_HEIGHT_HDC_EXCIT_H_WRAP = list(range(self.YAW_HEIGHT_HDC_H_DIM - self.YAW_HEIGHT_HDC_EXCIT_H_DIM_HALF, self.YAW_HEIGHT_HDC_H_DIM)) + \
                               list(range(0, self.YAW_HEIGHT_HDC_H_DIM)) + list(range(0, self.YAW_HEIGHT_HDC_EXCIT_H_DIM_HALF))

        self.YAW_HEIGHT_HDC_INHIB_Y_WRAP = list(range(self.YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF, self.YAW_HEIGHT_HDC_Y_DIM)) + \
                               list(range(0, self.YAW_HEIGHT_HDC_Y_DIM)) + list(range(0, self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF))
        self.YAW_HEIGHT_HDC_INHIB_H_WRAP = list(range(self.YAW_HEIGHT_HDC_H_DIM - self.YAW_HEIGHT_HDC_INHIB_H_DIM_HALF, self.YAW_HEIGHT_HDC_H_DIM)) + \
                               list(range(0, self.YAW_HEIGHT_HDC_H_DIM)) + list(range(0, self.YAW_HEIGHT_HDC_INHIB_H_DIM_HALF))

        #  The wrap for finding maximum activity packet
        self.YAW_HEIGHT_HDC_MAX_Y_WRAP = list(range(self.YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_PACKET_SIZE, self.YAW_HEIGHT_HDC_Y_DIM)) + \
                                list(range(0, self.YAW_HEIGHT_HDC_Y_DIM)) + list(range(0, self.YAW_HEIGHT_HDC_PACKET_SIZE))
        self.YAW_HEIGHT_HDC_MAX_H_WRAP = list(range(self.YAW_HEIGHT_HDC_H_DIM - self.YAW_HEIGHT_HDC_PACKET_SIZE + 1, self.YAW_HEIGHT_HDC_H_DIM)) + \
                                list(range(0, self.YAW_HEIGHT_HDC_H_DIM)) + list(range(0, self.YAW_HEIGHT_HDC_PACKET_SIZE))

        # 兴奋矩阵权重和抑制矩阵权重
        self.YAW_HEIGHT_HDC_EXCIT_WEIGHT = self.create_yaw_height_hdc_weights(self.YAW_HEIGHT_HDC_EXCIT_Y_DIM,
                            self.YAW_HEIGHT_HDC_EXCIT_H_DIM, self.YAW_HEIGHT_HDC_EXCIT_Y_VAR, self.YAW_HEIGHT_HDC_EXCIT_H_VAR)

        self.YAW_HEIGHT_HDC_INHIB_WEIGHT = self.create_yaw_height_hdc_weights(self.YAW_HEIGHT_HDC_INHIB_Y_DIM,
                            self.YAW_HEIGHT_HDC_INHIB_H_DIM, self.YAW_HEIGHT_HDC_INHIB_Y_VAR, self.YAW_HEIGHT_HDC_INHIB_Y_VAR)

        self.YAW_HEIGHT_HDC = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))
        curYawTheta, curHeight = self.get_hdc_initial_pos()
        self.YAW_HEIGHT_HDC[curYawTheta, curHeight] = 1                   # 设置初始值

        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = [curYawTheta, curHeight]

    # ***********************************************************
    # set the initial position in the hdcell network
    def get_hdc_initial_pos(self):
        curYawTheta = 1
        curHeight = 1
        return curYawTheta, curHeight

    # ***********************************************************
    # 创建权重矩阵
    def create_yaw_height_hdc_weights(self, yawDim, heightDim, yawVar, heightVar):

        yawDimCentre = math.floor(yawDim / 2) + 1
        heightDimCentre = math.floor(heightDim / 2) + 1
        weight = np.zeros((yawDim, heightDim))

        for h in range(0, heightDim):
            for y in range(0, yawDim):
                    weight[y, h] = 1.0 / (yawVar * np.sqrt(2 * np.pi)) * np.exp(
                        (-(y - yawDimCentre) ** 2) / (2 * yawVar ** 2)) * \
                                      1.0 / (heightVar * np.sqrt(2 * np.pi)) * np.exp(
                        (-(h - heightDimCentre) ** 2) / (2 * heightVar ** 2))

        # 归一化
        weight /= np.sum(weight)

        return weight

    # ***********************************************************
    #
    def get_current_yaw_height_value(self):

        # find the max activated cell
        (y, h) = np.unravel_index(self.YAW_HEIGHT_HDC.argmax(), self.YAW_HEIGHT_HDC.shape)  # 返回最大值所在的下标

        # take the max activated cell +- AVG_CELL in 2d space
        tempYawHeightHdc = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))

        tempYawHeightHdc[np.ix_(self.YAW_HEIGHT_HDC_MAX_Y_WRAP[y:y + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2 + 1],
                             self.YAW_HEIGHT_HDC_MAX_H_WRAP[h:h + self. YAW_HEIGHT_HDC_PACKET_SIZE * 2 + 1])] = \
            self.YAW_HEIGHT_HDC[np.ix_(self.YAW_HEIGHT_HDC_MAX_Y_WRAP[y:y + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2 + 1],
                                  self.YAW_HEIGHT_HDC_MAX_H_WRAP[h:h + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2 + 1])]

        yawSumSin = np.sum(np.dot(self.YAW_HEIGHT_HDC_Y_SUM_SIN_LOOKUP, np.sum(tempYawHeightHdc, 1)))
        yawSumCos = np.sum(np.dot(self.YAW_HEIGHT_HDC_Y_SUM_COS_LOOKUP, np.sum(tempYawHeightHdc, 1)))

        heightSumSin = np.sum(np.dot(self.YAW_HEIGHT_HDC_H_SUM_SIN_LOOKUP, np.sum(tempYawHeightHdc, 1)))
        heightSumCos = np.sum(np.dot(self.YAW_HEIGHT_HDC_H_SUM_COS_LOOKUP, np.sum(tempYawHeightHdc, 1)))

        outYawTheta = np.mod(np.arctan2(yawSumSin, yawSumCos) / self.YAW_HEIGHT_HDC_Y_TH_SIZE, self.YAW_HEIGHT_HDC_Y_DIM)
        outHeightValue = np.mod(np.arctan2(heightSumSin, heightSumCos) / self.YAW_HEIGHT_HDC_H_SIZE, self.YAW_HEIGHT_HDC_H_DIM)

        return outYawTheta, outHeightValue

    # ***********************************************************
    def yaw_height_hdc_iteration(self, ododelta, v_temp):

        transV = ododelta[0]
        yawRotV = ododelta[1]
        heightV = ododelta[2]

        # 注入能量
        if (v_temp.first != True):
            act_yaw = min(max(round(v_temp.hdc_yaw), 1), self.YAW_HEIGHT_HDC_Y_DIM)
            act_height = min(max(round(v_temp.hdc_height), 1), self.YAW_HEIGHT_HDC_H_DIM)
            print(act_yaw)

            energy = self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY * 1.0 / 30.0 * \
                     (30.0 - np.exp(1.2 * 1.0))                 ########?????????????????
            if energy > 0:
                self.YAW_HEIGHT_HDC[act_yaw, act_height] += energy

        #  Local excitation: yaw_height_hdc_local_excitation = yaw_height_hdc elements * yaw_height_hdc weights
        yaw_height_hdc_local_excit_new = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))

        for h in range(0, self.YAW_HEIGHT_HDC_H_DIM):
            for y in range(0, self.YAW_HEIGHT_HDC_Y_DIM):
                    if self.YAW_HEIGHT_HDC[y, h] != 0:
                        yaw_height_hdc_local_excit_new[np.ix_(self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP[y:y + self.YAW_HEIGHT_HDC_EXCIT_Y_DIM],
                                                        self.YAW_HEIGHT_HDC_EXCIT_H_WRAP[h:h + self.YAW_HEIGHT_HDC_EXCIT_H_DIM])] += \
                            self.YAW_HEIGHT_HDC[y, h] * self.YAW_HEIGHT_HDC_EXCIT_WEIGHT

        self.YAW_HEIGHT_HDC = yaw_height_hdc_local_excit_new

        # local inhibition: yaw_height_hdc_local_inhibition = hdc - hdc elements * hdc_inhib weights
        yaw_height_hdc_local_inhib_new = np.zeros((self.YAW_HEIGHT_HDC_Y_DIM, self.YAW_HEIGHT_HDC_H_DIM))
        for h in range(0, self.YAW_HEIGHT_HDC_H_DIM):
            for y in range(0, self.YAW_HEIGHT_HDC_Y_DIM):
                    if self.YAW_HEIGHT_HDC[y, h] != 0:
                        yaw_height_hdc_local_inhib_new[np.ix_(self.YAW_HEIGHT_HDC_INHIB_Y_WRAP[y:y + self.YAW_HEIGHT_HDC_INHIB_Y_DIM],
                                                        self.YAW_HEIGHT_HDC_INHIB_H_WRAP[h:h + self. YAW_HEIGHT_HDC_INHIB_H_DIM])] += \
                            self.YAW_HEIGHT_HDC[y, h] * self.YAW_HEIGHT_HDC_INHIB_WEIGHT

        self.YAW_HEIGHT_HDC -= yaw_height_hdc_local_inhib_new

        # global inhibition - PC_gi = PC_li elements - inhibition
        self.YAW_HEIGHT_HDC[self.YAW_HEIGHT_HDC < self.YAW_HEIGHT_HDC_GLOBAL_INHIB] = 0
        self.YAW_HEIGHT_HDC[self.YAW_HEIGHT_HDC >= self.YAW_HEIGHT_HDC_GLOBAL_INHIB] -= self.YAW_HEIGHT_HDC_GLOBAL_INHIB

        # normalisation
        total = np.sum(self.YAW_HEIGHT_HDC)
        self.YAW_HEIGHT_HDC /= total

        if yawRotV != 0:
            weight = np.mod(abs(yawRotV)/self.YAW_HEIGHT_HDC_Y_TH_SIZE, 1)
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(yawRotV) * math.floor(np.mod(abs(yawRotV) / self.YAW_HEIGHT_HDC_Y_TH_SIZE, self.YAW_HEIGHT_HDC_Y_DIM)))
            shift2 = int(np.sign(yawRotV) * math.ceil(np.mod(abs(yawRotV) / self.YAW_HEIGHT_HDC_Y_TH_SIZE, self.YAW_HEIGHT_HDC_Y_DIM)))
            self.YAW_HEIGHT_HDC = np.roll(self.YAW_HEIGHT_HDC, shift1, axis=0) * (1.0 - weight) + \
                             np.roll(self.YAW_HEIGHT_HDC, shift2, axis=0) * weight

        if heightV != 0:
            weight = np.mod(abs(heightV)/self.YAW_HEIGHT_HDC_H_SIZE, 1)
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(heightV) * math.floor(np.mod(abs(heightV) / self.YAW_HEIGHT_HDC_H_SIZE, self.YAW_HEIGHT_HDC_H_DIM)))
            shift2 = int(np.sign(heightV) * math.ceil(np.mod(abs(heightV) / self.YAW_HEIGHT_HDC_H_SIZE, self.YAW_HEIGHT_HDC_H_DIM)))
            self.YAW_HEIGHT_HDC = np.roll(self.YAW_HEIGHT_HDC, shift1, axis=1) * (1.0 - weight) + \
                             np.roll(self.YAW_HEIGHT_HDC, shift2, axis=1) * weight

        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = self.get_current_yaw_height_value()
        return self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH