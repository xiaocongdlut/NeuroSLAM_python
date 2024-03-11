# -*- coding: utf-8 -*-
# @Time : 2023/3/2 14:28
# @Author : xiao cong
# @Description :

import numpy as np
import logging


# Clip the input angle to between -pi and pi radians
def clip_radian_180(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


# Clip the input angle to between 0 and 2pi radians
def clip_radian_360(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle


# Get the minimum delta distance between two values assuming a wrap to zero at max
def get_min_delta(d1, d2, max):
    delta = min([abs(d1 - d2), max - abs(d1 - d2)])
    return delta


# Get the signed delta angle from angle1 to angle2 handling the wrap from 2pi to 0
def signed_delta_rad(angle1, angle2):
    dir = clip_radian_180(angle2 - angle1)

    delta_angle = abs(clip_radian_360(angle1) - clip_radian_360(angle2))

    if delta_angle < (2 * np.pi - delta_angle):
        if dir > 0:
            angle = delta_angle
        else:
            angle = -delta_angle
    else:
        if dir > 0:
            angle = 2 * np.pi - delta_angle
        else:
            angle = -(2 * np.pi - delta_angle)
    return angle


class Experience:
    "A point in the experience map"

    def __init__(self, parent, xGc, yGc, zGc, curYawHdc, curHeight, vt, x_exp, y_exp, z_exp, yaw_exp_rad):
        self.xGc = xGc  # 网络更新后的激活细胞位置
        self.yGc = yGc
        self.zGc = zGc
        self.curYawHdc = curYawHdc
        self.curHeight = curHeight

        self.x_exp = x_exp  # x_exp is xcoordinate in experience map
        self.y_exp = y_exp  # y_exp is ycoordinate in experience map
        self.z_exp = z_exp  # y_exp is ycoordinate in experience map

        self.vt = vt
        self.yaw_exp_rad = yaw_exp_rad
        self.parent_map = parent  # ExperienceMap 类数据

        self.current_exp = None

        self.links = []

    def link_to(self, target):
        pm = self.parent_map
        d_xy = np.sqrt(pm.ACCUM_DELTA_X ** 2 + pm.ACCUM_DELTA_Y ** 2)
        d_z = pm.ACCUM_DELTA_Z
        heading_yaw_exp_rad = signed_delta_rad(self.yaw_exp_rad,
                                               -np.arctan2(pm.ACCUM_DELTA_Y, pm.ACCUM_DELTA_X))
        facing_yaw_exp_rad = signed_delta_rad(self.yaw_exp_rad, pm.ACCUM_DELTA_YAW)

        new_link = ExperienceLink(target, heading_yaw_exp_rad, facing_yaw_exp_rad, d_xy, d_z)
        self.links.append(new_link)


class ExperienceLink:
    "A directed link between two Experience objects"

    def __init__(self, target, heading_yaw_exp_rad, facing_yaw_exp_rad, d_xy, d_z):
        self.target = target
        self.heading_yaw_exp_rad = heading_yaw_exp_rad
        self.facing_yaw_exp_rad = facing_yaw_exp_rad
        self.d_xy = d_xy
        self.d_z = d_z


class ExperienceMap:
    def __init__(self, **kwargs):

        self.GC_X_DIM = kwargs.pop("GC_X_DIM", 36)
        self.GC_Y_DIM = kwargs.pop("GC_Y_DIM", 36)
        self.GC_Z_DIM = kwargs.pop("GC_Z_DIM", 36)
        self.YAW_HEIGHT_HDC_Y_DIM = kwargs.pop("YAW_HEIGHT_HDC_Y_DIM", 36)
        self.YAW_HEIGHT_HDC_H_DIM = kwargs.pop("YAW_HEIGHT_HDC_H_DIM", 36)

        self.DELTA_EXP_GC_HDC_THRESHOLD = kwargs.pop("DELTA_EXP_GC_HDC_THRESHOLD", 45)  # 形成新经验的阈值
                                                                                # 40
        self.EXP_LOOPS = kwargs.pop("EXP_LOOPS", 1)
        self.EXP_CORRECTION = kwargs.pop("EXP_CORRECTION", 0.1)

        self.exps = []
        self.current_exp = None

        self.ACCUM_DELTA_X = 0
        self.ACCUM_DELTA_Y = 0
        self.ACCUM_DELTA_Z = 0
        self.ACCUM_DELTA_YAW = 0

        self.current_vt = None
        self.history = []

    def create_and_link_new_exp(self, vt, xGc, yGc, zGc, curYawHdc, curHeight):
        "Create a new experience and link current experience to it"

        new_x_exp = self.ACCUM_DELTA_X
        new_y_exp = self.ACCUM_DELTA_Y
        new_z_exp = self.ACCUM_DELTA_Z
        new_yaw_exp_rad = clip_radian_180(self.ACCUM_DELTA_YAW)

        # add contributions from the current exp, if one is available
        if self.current_exp is not None:
            new_x_exp += self.current_exp.x_exp
            new_y_exp += self.current_exp.y_exp
            new_z_exp += self.current_exp.z_exp
            new_yaw_exp_rad = clip_radian_180(self.ACCUM_DELTA_YAW)

        # create a new expereince
        new_exp = Experience(self, xGc, yGc, zGc, curYawHdc, curHeight, vt, new_x_exp, new_y_exp, new_z_exp, new_yaw_exp_rad)

        if self.current_exp is not None:
            # link the current_exp for this one
            self.current_exp.link_to(new_exp)

        # add the new experience to the map
        self.exps.append(new_exp)

        # add this experience to the view template for efficient lookup
        vt.exps.append(new_exp)

        return new_exp

    def update(self, transV, yawRotV, heightV, xGc, yGc, zGc, curYawHdc, curHeight, vt):
        """ Update the experience map

            (Matlab version: equivalent to rs_experience_map_iteration)
        """

        # update accumulators
        self.ACCUM_DELTA_YAW = clip_radian_180(self.ACCUM_DELTA_YAW + yawRotV)
        self.ACCUM_DELTA_X += transV * np.cos(self.ACCUM_DELTA_YAW)
        self.ACCUM_DELTA_Y += transV * np.sin(self.ACCUM_DELTA_YAW)
        self.ACCUM_DELTA_Z += heightV

        # check if this the first update
        if self.current_exp is None:
            # first experience
            delta_em = 0
        else:
            # subsequent experience

            minDeltaX = get_min_delta(self.current_exp.xGc, xGc, self.GC_X_DIM)
            minDeltaY = get_min_delta(self.current_exp.yGc, yGc, self.GC_Y_DIM)
            minDeltaZ = get_min_delta(self.current_exp.zGc, zGc, self.GC_Z_DIM)

            minDeltaYaw = get_min_delta(self.current_exp.curYawHdc, curYawHdc, self.YAW_HEIGHT_HDC_Y_DIM)
            minDeltaHeight = get_min_delta(self.current_exp.curHeight, curHeight, self.YAW_HEIGHT_HDC_H_DIM)
            minDeltaYawReversed = get_min_delta(self.current_exp.curYawHdc, (self.YAW_HEIGHT_HDC_Y_DIM / 2) - curYawHdc,
                                                self.YAW_HEIGHT_HDC_Y_DIM)

            minDeltaYaw = min(minDeltaYaw, minDeltaYawReversed)
            delta_em = np.sqrt(
                minDeltaX ** 2 + minDeltaY ** 2 + minDeltaZ ** 2 + minDeltaYaw ** 2 + minDeltaHeight ** 2)
            print(delta_em)
        adjust_map = False
        print(len(vt.exps))
        # if this view template is not associated with any experiences yet,
        # or if the pc x,y,th has changed enough, create and link a new exp
        if len(vt.exps) == 0 or (delta_em > self.DELTA_EXP_GC_HDC_THRESHOLD):
            new_exp = self.create_and_link_new_exp(vt, xGc, yGc, zGc, curYawHdc, curHeight)
            self.current_exp = new_exp
            print("lll")
            # reset accumulators
            self.ACCUM_DELTA_X = 0
            self.ACCUM_DELTA_Y = 0
            self.ACCUM_DELTA_Z = 0
            self.ACCUM_DELTA_YAW = self.current_exp.yaw_exp_rad

            # print(vt)
            # print(self.current_exp.vt)

        # if the view template has changed (but isn't new) search for the
        # mathcing experience
        elif vt != self.current_exp.vt:           # ??????????????????? 此处有问题

            # find the exp associated with the current vt and that is under the
            # threshold distance to the centre of pose cell activity
            # if multiple exps are under the threshold then don't match (to
            # reduce hash collisions)
            print("ddd")
            adjust_map = True
            matched_exp = None
            delta_ems = []
            n_candidate_matches = 0
            for (i, e) in enumerate(vt.exps):

                minDeltaYaw = get_min_delta(e.curYawHdc, curYawHdc, self.YAW_HEIGHT_HDC_Y_DIM)
                minDeltaHeight = get_min_delta(e.curHeight, curHeight, self.YAW_HEIGHT_HDC_Y_DIM)
                delta_em = np.sqrt(get_min_delta(e.xGc, xGc, self.GC_X_DIM) ** 2 \
                                   + get_min_delta(e.yGc, yGc, self.GC_Y_DIM) ** 2 \
                                   + get_min_delta(e.zGc, zGc, self.GC_Z_DIM) ** 2
                                   + minDeltaYaw ** 2 + minDeltaHeight ** 2)

                delta_ems.append(delta_em)

                if delta_em < self.DELTA_EXP_GC_HDC_THRESHOLD:
                    n_candidate_matches += 1

            if n_candidate_matches > 1:
                # this means we aren't sure about which experience is a match
                # due to hash table collision instead of a false posivitive
                # which may create blunder links in the experience map keep
                # the previous experience matched_exp_count

                # TODO: raise?
                # TODO: what is being accomplished here
                #       check rs_experience_map_iteration.m, line 75-ish
                logging.warning("Too many candidate matches when searching" + \
                                " experience map")
                pass
            else:
                min_delta_val = min(delta_ems)
                min_delta_id = delta_ems.index(min_delta_val)

                if min_delta_val < self.DELTA_EXP_GC_HDC_THRESHOLD:
                    matched_exp = vt.exps[min_delta_id]

                    # check if a link already exists
                    link_exists = False

                    for linked_exp in [l.target for l in self.current_exp.links]:
                        if linked_exp == matched_exp:
                            link_exists = True
                            break

                    if not link_exists:
                        self.current_exp.link_to(matched_exp)

                # self.exp_id = len(self.exps)-1

                if matched_exp is None:
                    matched_exp = self.create_and_link_new_exp(vt, xGc, yGc, zGc, curYawHdc, curHeight)

                self.ACCUM_DELTA_X = 0
                self.ACCUM_DELTA_Y = 0
                self.ACCUM_DELTA_Z = 0
                self.ACCUM_DELTA_YAW = self.current_exp.yaw_exp_rad
                self.current_exp = matched_exp

        self.history.append(self.current_exp)

        if not adjust_map:
            return
        # return

        # Iteratively update the experience map with the new information
        for i in range(0, self.EXP_LOOPS):
            for e0 in self.exps:
                for l in e0.links:
                    # e0 is the experience under consideration
                    # e1 is an experience linked from e0
                    # l is the link object which contains additoinal heading
                    # info
                    print("jjj")
                    e1 = l.target

                    # work out where exp0 thinks exp1 (x,y) should be based on
                    # the stored link information
                    lx = e0.x_exp + l.d_xy * np.cos(e0.yaw_exp_rad + l.heading_yaw_exp_rad)
                    ly = e0.y_exp + l.d_xy * np.sin(e0.yaw_exp_rad + l.heading_yaw_exp_rad)
                    lz = e0.z_exp + l.d_z

                    # correct e0 and e1 (x,y) by equal but opposite amounts
                    # a 0.5 correction parameter means that e0 and e1 will be
                    # fully corrected based on e0's link information
                    e0.x_exp += (e1.x_exp - lx) * self.EXP_CORRECTION
                    e0.y_exp += (e1.y_exp - ly) * self.EXP_CORRECTION
                    e0.z_exp += (e1.z_exp - lz) * self.EXP_CORRECTION

                    e1.x_exp -= (e1.x_exp - lx) * self.EXP_CORRECTION
                    e1.y_exp -= (e1.y_exp - ly) * self.EXP_CORRECTION
                    e1.z_exp -= (e1.z_exp - lz) * self.EXP_CORRECTION

                    # determine the angle between where e0 thinks e1's facing
                    # should be based on the link information
                    TempDeltaYawFacing = signed_delta_rad(e0.yaw_exp_rad + l.facing_yaw_exp_rad, e1.yaw_exp_rad)

                    # correct e0 and e1 facing by equal but opposite amounts
                    # a 0.5 correction parameter means that e0 and e1 will be
                    # fully corrected based on e0's link information
                    e0.yaw_exp_rad = clip_radian_180(e0.yaw_exp_rad + TempDeltaYawFacing * self.EXP_CORRECTION)
                    e1.yaw_exp_rad = clip_radian_180(e1.yaw_exp_rad - TempDeltaYawFacing * self.EXP_CORRECTION)

        return
