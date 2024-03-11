# -*- coding: utf-8 -*-
# @Time : 2023/2/24 15:30
# @Author : xiao cong
# @Description :

import numpy as np
import math

global VT


class GridCellsNetwork:

    # ***********************************************************
    # 定义基本参数
    def __init__(self, **kwargs):

        # 网格细胞维度
        self.GC_X_DIM = kwargs.pop("GC_X_DIM", 36)
        self.GC_Y_DIM = kwargs.pop("GC_Y_DIM", 36)
        self.GC_Z_DIM = kwargs.pop("GC_Z_DIM", 36)

        # 局部兴奋权重矩阵的维度
        self.GC_EXCIT_X_DIM = kwargs.pop("GC_EXCIT_X_DIM", 7)
        self.GC_EXCIT_Y_DIM = kwargs.pop("GC_EXCIT_Y_DIM", 7)
        self.GC_EXCIT_Z_DIM = kwargs.pop("GC_EXCIT_Z_DIM", 7)

        # 局部抑制权重矩阵的维度
        self.GC_INHIB_X_DIM = kwargs.pop("GC_INHIB_X_DIM", 5)
        self.GC_INHIB_Y_DIM = kwargs.pop("GC_INHIB_Y_DIM", 5)
        self.GC_INHIB_Z_DIM = kwargs.pop("GC_INHIB_Z_DIM", 5)

        # 全局抑制的值
        self.GC_GLOBAL_INHIB = kwargs.pop("GC_GLOBAL_INHIB", 0.0002)

        # 当模板匹配时，注入的能量
        self.GC_VT_INJECT_ENERGY = kwargs.pop("GC_VT_INJECT_ENERGY", 0.1)

        # 方差
        self.GC_EXCIT_X_VAR = kwargs.pop("GC_EXCIT_X_VAR", 1.5)
        self.GC_EXCIT_Y_VAR = kwargs.pop("GC_EXCIT_Y_VAR", 1.5)
        self.GC_EXCIT_Z_VAR = kwargs.pop("GC_EXCIT_Z_VAR", 1.5)

        self.GC_INHIB_X_VAR = kwargs.pop("GC_INHIB_X_VAR", 2)
        self.GC_INHIB_Y_VAR = kwargs.pop("GC_INHIB_Y_VAR", 2)
        self.GC_INHIB_Z_VAR = kwargs.pop("GC_INHIB_Z_VAR", 2)

        # The scale of horizontal translational velocity
        self.GC_HORI_TRANS_V_SCALE = kwargs.pop("GC_HORI_TRANS_V_SCALE", 1)

        # The scale of vertical translational velocity
        self.GC_VERT_TRANS_V_SCALE = kwargs.pop("GC_VERT_TRANS_V_SCALE", 1)

        # 活动包大小
        self.GC_PACKET_SIZE = kwargs.pop("GC_PACKET_SIZE", 4)

        self.GC_EXCIT_X_DIM_HALF = math.floor(self.GC_EXCIT_X_DIM / 2)
        self.GC_EXCIT_Y_DIM_HALF = math.floor(self.GC_EXCIT_Y_DIM / 2)
        self.GC_EXCIT_Z_DIM_HALF = math.floor(self.GC_EXCIT_Z_DIM / 2)

        self.GC_INHIB_X_DIM_HALF = math.floor(self.GC_INHIB_X_DIM / 2)
        self.GC_INHIB_Y_DIM_HALF = math.floor(self.GC_INHIB_Y_DIM / 2)
        self.GC_INHIB_Z_DIM_HALF = math.floor(self.GC_INHIB_Z_DIM / 2)

        self.GC_EXCIT_X_WRAP = list(range(self.GC_X_DIM - self.GC_EXCIT_X_DIM_HALF, self.GC_X_DIM)) + \
                               list(range(0, self.GC_X_DIM)) + list(range(0, self.GC_EXCIT_X_DIM_HALF))
        self.GC_EXCIT_Y_WRAP = list(range(self.GC_Y_DIM - self.GC_EXCIT_Y_DIM_HALF, self.GC_Y_DIM)) + \
                               list(range(0, self.GC_Y_DIM)) + list(range(0, self.GC_EXCIT_Y_DIM_HALF))
        self.GC_EXCIT_Z_WRAP = list(range(self.GC_Z_DIM - self.GC_EXCIT_Z_DIM_HALF, self.GC_Z_DIM)) + \
                               list(range(0, self.GC_Z_DIM)) + list(range(0, self.GC_EXCIT_Z_DIM_HALF))

        self.GC_INHIB_X_WRAP = list(range(self.GC_X_DIM - self.GC_INHIB_X_DIM_HALF, self.GC_X_DIM)) + \
                               list(range(0, self.GC_X_DIM)) + list(range(0, self.GC_INHIB_X_DIM_HALF))
        self.GC_INHIB_Y_WRAP = list(range(self.GC_Y_DIM - self.GC_INHIB_Y_DIM_HALF, self.GC_Y_DIM)) + \
                               list(range(0, self.GC_Y_DIM)) + list(range(0, self.GC_INHIB_Y_DIM_HALF))
        self.GC_INHIB_Z_WRAP = list(range(self.GC_Z_DIM - self.GC_INHIB_Z_DIM_HALF, self.GC_Z_DIM)) + \
                               list(range(0, self.GC_Z_DIM)) + list(range(0, self.GC_INHIB_Z_DIM_HALF))

        # 每个细胞表示的角度
        self.GC_X_TH_SIZE = 2 * np.pi / self.GC_X_DIM
        self.GC_Y_TH_SIZE = 2 * np.pi / self.GC_Y_DIM
        self.GC_Z_TH_SIZE = 2 * np.pi / self.GC_Y_DIM

        self.GC_X_SUM_SIN_LOOKUP = np.sin(np.arange(1, self.GC_X_DIM + 1) * self.GC_X_TH_SIZE)
        self.GC_X_SUM_COS_LOOKUP = np.cos(np.arange(1, self.GC_X_DIM + 1) * self.GC_X_TH_SIZE)

        self.GC_Y_SUM_SIN_LOOKUP = np.sin(np.arange(1, self.GC_Y_DIM + 1) * self.GC_Y_TH_SIZE)
        self.GC_Y_SUM_COS_LOOKUP = np.cos(np.arange(1, self.GC_Y_DIM + 1) * self.GC_Y_TH_SIZE)

        self.GC_Z_SUM_SIN_LOOKUP = np.sin(np.arange(1, self.GC_Z_DIM + 1) * self.GC_Z_TH_SIZE)
        self.GC_Z_SUM_COS_LOOKUP = np.cos(np.arange(1, self.GC_Z_DIM + 1) * self.GC_Z_TH_SIZE)

        # The wrap for finding maximum activity packet
        self.GC_MAX_X_WRAP = list(range(self.GC_X_DIM - self.GC_PACKET_SIZE, self.GC_X_DIM)) + \
                             list(range(0, self.GC_X_DIM)) + list(range(0, self.GC_PACKET_SIZE))
        self.GC_MAX_Y_WRAP = list(range(self.GC_Y_DIM - self.GC_PACKET_SIZE, self.GC_Y_DIM)) + \
                             list(range(0, self.GC_Y_DIM)) + list(range(0, self.GC_PACKET_SIZE))
        self.GC_MAX_Z_WRAP = list(range(self.GC_Y_DIM - self.GC_PACKET_SIZE, self.GC_Y_DIM)) + \
                             list(range(0, self.GC_Y_DIM)) + list(range(0, self.GC_PACKET_SIZE))


        # 兴奋矩阵权重和抑制矩阵权重
        self.GC_EXCIT_WEIGHT = self.create_gc_weights(self.GC_EXCIT_X_DIM, self.GC_EXCIT_Y_DIM, self.GC_EXCIT_Z_DIM,
                                                      self.GC_EXCIT_X_VAR, self.GC_EXCIT_Y_VAR, self.GC_EXCIT_Z_VAR)
        self.GC_INHIB_WEIGHT = self.create_gc_weights(self.GC_INHIB_X_DIM, self.GC_INHIB_Y_DIM, self.GC_INHIB_Z_DIM,
                                                      self.GC_INHIB_X_VAR, self.GC_INHIB_X_VAR, self.GC_INHIB_Z_VAR)

        self.GRIDCELLS = np.zeros((self.GC_X_DIM, self.GC_Y_DIM, self.GC_Z_DIM))
        gcX, gcY, gcZ = self.get_gc_initial_pos()      # 设置初始值
        self.GRIDCELLS[gcX, gcY, gcZ] = 1

        self.MAX_ACTIVE_XYZ_PATH = [gcX, gcY, gcZ]

    # ***********************************************************
    # set the initial position in the grid cell network
    def get_gc_initial_pos(self):

        gcX = math.floor(self.GC_X_DIM / 2)
        gcY = math.floor(self.GC_Y_DIM / 2)
        gcZ = math.floor(self.GC_Z_DIM / 2)

        return gcX, gcY, gcZ

    # ***********************************************************
    # 创建权重矩阵
    def create_gc_weights(self, xDim, yDim, zDim, xVar, yVar, zVar):

        xDimCentre = math.floor(xDim + 1)
        yDimCentre = math.floor(yDim + 1)
        zDimCentre = math.floor(zDim + 1)
        weight = np.zeros((xDim, yDim, zDim))

        for x in range(0, xDim):
            for y in range(0, yDim):
                for z in range(0, zDim):
                    weight[x, y, z] = 1.0 / (xVar * np.sqrt(2 * np.pi)) * np.exp(
                        (-(x - xDimCentre) ** 2) / (2 * xVar ** 2)) * \
                                      1.0 / (yVar * np.sqrt(2 * np.pi)) * np.exp(
                        (-(y - yDimCentre) ** 2) / (2 * yVar ** 2)) * \
                                      1.0 / (zVar * np.sqrt(2 * np.pi)) * np.exp(
                        (-(z - zDimCentre) ** 2) / (2 * zVar ** 2))

        # 归一化
        weight /= np.sum(weight)

        return weight

    # ***********************************************************
    #
    def get_gc_xyz(self):

        # find the max activated cell
        (x, y, z) = np.unravel_index(self.GRIDCELLS.argmax(), self.GRIDCELLS.shape)  # 返回最大值所在的下标

        # take the max activated cell + - AVG_CELL in 3d space
        tempGridcells = np.zeros((self.GC_X_DIM, self.GC_X_DIM, self.GC_Z_DIM))

        tempGridcells[np.ix_(self.GC_MAX_X_WRAP[x:x + self.GC_PACKET_SIZE * 2 + 1],
                             self.GC_MAX_Y_WRAP[y:y + self.GC_PACKET_SIZE * 2 + 1],
                             self.GC_MAX_Z_WRAP[z:z + self.GC_PACKET_SIZE * 2 + 1])] = \
            self.GRIDCELLS[np.ix_(self.GC_MAX_X_WRAP[x:x + self.GC_PACKET_SIZE * 2 + 1],
                                  self.GC_MAX_Y_WRAP[y:y + self.GC_PACKET_SIZE * 2 + 1],
                                  self.GC_MAX_Z_WRAP[z:z + self.GC_PACKET_SIZE * 2 + 1])]

        xSumSin = np.sum(np.dot(self.GC_X_SUM_SIN_LOOKUP, np.sum(np.sum(tempGridcells, 1), 1)))     # ????
        xSumCos = np.sum(np.dot(self.GC_X_SUM_COS_LOOKUP, np.sum(np.sum(tempGridcells, 1), 1)))
        ySumSin = np.sum(np.dot(self.GC_Y_SUM_SIN_LOOKUP, np.sum(np.sum(tempGridcells, 0), 1)))
        ySumCos = np.sum(np.dot(self.GC_Y_SUM_COS_LOOKUP, np.sum(np.sum(tempGridcells, 0), 1)))
        zSumSin = np.sum(np.dot(self.GC_Z_SUM_SIN_LOOKUP, np.sum(np.sum(tempGridcells, 0), 0)))
        zSumCos = np.sum(np.dot(self.GC_Z_SUM_COS_LOOKUP, np.sum(np.sum(tempGridcells, 0), 0)))

        gcX = np.mod(np.arctan2(xSumSin, xSumCos) / self.GC_X_TH_SIZE, self.GC_X_DIM)
        gcY = np.mod(np.arctan2(ySumSin, ySumCos) / self.GC_Y_TH_SIZE, self.GC_Y_DIM)
        gcZ = np.mod(np.arctan2(zSumSin, zSumCos) / self.GC_Z_TH_SIZE, self.GC_Z_DIM)

        return gcX, gcY, gcZ

    # ***********************************************************
    # 更新迭代
    def gc_iteration(self, ododelta, v_temp):

        transV = ododelta[0]                     #### × scale ???
        yawRotV = ododelta[1]
        heightV = ododelta[2]

        # 注入能量
        if (v_temp.first != True):
            actX = min(max(round(v_temp.x_pc), 1), self.GC_X_DIM)
            actY = min(max(round(v_temp.y_pc), 1), self.GC_Y_DIM)
            actZ = min(max(round(v_temp.z_pc), 1), self.GC_Z_DIM)

            energy = self.GC_VT_INJECT_ENERGY * 1.0 / 30.0 * \
                     (30.0 - np.exp(1.2 * v_temp.decay))
            if energy > 0:
                self.GRIDCELLS[actX, actY, actZ] += energy

        # 局部兴奋
        # local excitation GC_local_excitation = GC elements * GC weights
        gridcell_local_excit_new = np.zeros((self.GC_X_DIM, self.GC_X_DIM, self.GC_Z_DIM))

        for x in range(0, self.GC_X_DIM):
            for y in range(0, self.GC_Y_DIM):
                for z in range(0, self.GC_Z_DIM):
                    if self.GRIDCELLS[x, y, z] != 0:
                        gridcell_local_excit_new[np.ix_(self.GC_EXCIT_X_WRAP[x:x + self.GC_EXCIT_X_DIM],
                                                        self.GC_EXCIT_Y_WRAP[y:y + self.GC_EXCIT_Y_DIM],
                                                        self.GC_EXCIT_Z_WRAP[z:z + self.GC_EXCIT_Z_DIM])] += \
                            self.GRIDCELLS[x, y, z] * self.GC_EXCIT_WEIGHT

        self.GRIDCELLS = gridcell_local_excit_new

        # 局部抑制
        # local inhibition - GC_li = GC_le - GC_le elements * GC weights
        gridcell_local_inhib_new = np.zeros((self.GC_X_DIM, self.GC_X_DIM, self.GC_Z_DIM))

        for x in range(0, self.GC_X_DIM):
            for y in range(0, self.GC_Y_DIM):
                for z in range(0, self.GC_Z_DIM):
                    if self.GRIDCELLS[x, y, z] != 0:
                        gridcell_local_inhib_new[np.ix_(self.GC_INHIB_X_WRAP[x:x + self.GC_INHIB_X_DIM],
                                                        self.GC_INHIB_Y_WRAP[y:y + self.GC_INHIB_Y_DIM],
                                                        self.GC_INHIB_Z_WRAP[z:z + self.GC_INHIB_Z_DIM])] -= \
                            self.GRIDCELLS[x, y, z] * self.GC_INHIB_WEIGHT

        self.GRIDCELLS -= gridcell_local_inhib_new

        # 全局抑制
        # global inhibition - gc_gi = GC_lielements - inhibition
        self.GRIDCELLS[self.GRIDCELLS < self.GC_GLOBAL_INHIB] = 0
        self.GRIDCELLS[self.GRIDCELLS >= self.GC_GLOBAL_INHIB] -= self.GC_GLOBAL_INHIB

        # 正则化
        total = np.sum(self.GRIDCELLS)
        self.GRIDCELLS /= total

        # 路径积分
        for z in range(0, self.GC_Z_DIM):

            curYawThetaInRadian = float(z - 1) *(2*np.pi/36)             #####?????
            if curYawThetaInRadian == 0:
                self.GRIDCELLS[:, :, z] = self.GRIDCELLS[:, :, z] * (1.0 - transV) + \
                                          np.roll(self.GRIDCELLS[:, :, z], 1, 1) * transV
            elif curYawThetaInRadian == np.pi / 2:
                self.GRIDCELLS[:, :, z] = self.GRIDCELLS[:, :, z] * (1.0 - transV) + \
                                          np.roll(self.GRIDCELLS[:, :, z], 1, 0) * transV
            elif curYawThetaInRadian == np.pi:
                self.GRIDCELLS[:, :, z] = self.GRIDCELLS[:, :, z] * (1.0 - transV) + \
                                          np.roll(self.GRIDCELLS[:, :, z], -1, 1) * transV
            elif curYawThetaInRadian == 3 * np.pi / 2:
                self.GRIDCELLS[:, :, z] = self.GRIDCELLS[:, :, z] * (1.0 - transV) + \
                                          np.roll(self.GRIDCELLS[:, :, z], -1, 0) * transV
            else:
                gcInZPlane90 = np.rot90(self.GRIDCELLS[:, :, z], math.floor(curYawThetaInRadian * 2 / np.pi))

                dir90 = curYawThetaInRadian - math.floor(curYawThetaInRadian * 2 / np.pi) * np.pi / 2

                gcInZPlaneNew = np.zeros((self.GC_X_DIM + 2, self.GC_Y_DIM + 2))
                gcInZPlaneNew[1:-1, 1:-1] = gcInZPlane90

                weight_sw = (transV ** 2) * np.cos(dir90) * np.sin(dir90)
                weight_se = transV * np.sin(dir90) - (transV ** 2) * np.cos(dir90) * np.sin(dir90)
                weight_nw = transV * np.cos(dir90) - (transV ** 2) * np.cos(dir90) * np.sin(dir90)
                weight_ne = 1.0 - weight_sw - weight_se - weight_nw

                gcInZPlaneNew = gcInZPlaneNew * weight_ne + np.roll(gcInZPlaneNew, 1, 1) * weight_nw + \
                                np.roll(gcInZPlaneNew, 1, 0) * weight_se + np.roll(np.roll(gcInZPlaneNew, 1, 1), 1,
                                                                                   0) * weight_sw

                gcInZPlane90 = gcInZPlaneNew[1:-1, 1:-1]
                gcInZPlane90[1:, 0] = gcInZPlane90[1:, 0] + gcInZPlaneNew[2:-1, -1]
                gcInZPlane90[0, 1:] = gcInZPlane90[0, 1:] + gcInZPlaneNew[-1, 2:-1]
                gcInZPlane90[0, 0] = gcInZPlane90[0, 0] + gcInZPlaneNew[-1, -1]

                self.GRIDCELLS[:, :, z] = np.rot90(gcInZPlane90, 4 - math.floor(curYawThetaInRadian * 2 / np.pi))

        if heightV != 0:
            weight = np.mod(abs(heightV)/self.GC_Z_TH_SIZE, 1)
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(heightV) * math.floor(np.mod(abs(heightV) / self.GC_Z_TH_SIZE, self.GC_Z_DIM)))
            shift2 = int(np.sign(heightV) * math.ceil(np.mod(abs(heightV) / self.GC_Z_TH_SIZE, self.GC_Z_DIM)))
            self.GRIDCELLS = np.roll(self.GRIDCELLS, shift1, axis=2) * (1.0 - weight) + \
                             np.roll(self.GRIDCELLS, shift2, axis=2) * weight

        self.MAX_ACTIVE_XYZ_PATH = self.get_gc_xyz()

        return self.MAX_ACTIVE_XYZ_PATH