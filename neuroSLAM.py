# -*- coding: utf-8 -*-
# @Time : 2023/3/9 10:56
# @Author : xiao cong
# @Description :
import cv2

import grid_cells_network
import yaw_height_hdc_network
import multilayered_experience_map
import visual_odometry
import visual_template

from functools import partial

class neuroSLAM(object):

    def __init__(self, image_source_list):
        self.image_source_list = image_source_list

        gc_shape = [36, 36, 36]
        hdc_shape = [36, 36]

        self.view_templates = self.initialize_view_templates(gc_shape, hdc_shape)
        self.visual_odometer = self.initialize_odo()
        self.grid_cell_network = self.initialize_grid_cell_network()
        self.yaw_height_hdc_network = self.initialize_hdc_cell_network()
        self.experience_map = self.initialize_experience_map()

        self.time_step = 0

    # creating visual templatecollection and initializing it with the first image.
    def initialize_view_templates(self, gc_shape, hdc_shape):
        simple_template = partial(visual_template.IntensityProfileTemplate,
                                  VT_IMG_CROP_Y_RANGE=slice(0, 120),
                                  VT_IMG_CROP_X_RANGE=slice(0, 160))

        templates = visual_template.ViewTemplateCollection(simple_template,
                                                           global_decay=0.1,
                                                           match_threshold=0.22)

        im = cv2.imread(self.image_source_list[0], cv2.IMREAD_GRAYSCALE)


        # templates.update(zeros(561))

        templates.current_template = templates.match(im, gc_shape[0] / 2.0, gc_shape[1] / 2.0,
                                            gc_shape[2] / 2.0, hdc_shape[0]/2.0, hdc_shape[1]/2.0)

        return templates

    # initializing Visual Odometer
    def initialize_odo(self):
        im0 = cv2.imread(self.image_source_list[0], cv2.IMREAD_GRAYSCALE)
        vod = visual_odometry.VisualOdometry()
        vod.update(im0)
        return vod

    # initializing position of the grid cell network activity bubble at center
    def initialize_grid_cell_network(self):
        gc_net = grid_cells_network.GridCellsNetwork()

        v_temp = self.view_templates[0]
        gc_net.gc_iteration(self.visual_odometer.delta, v_temp)
        # pcmax = pcnet.get_pc_max(pcnet.avg_xywrap, pcnet.avg_thwrap)

        return gc_net

    # initializing position of the yaw-height cell network activity bubble at center
    def initialize_hdc_cell_network(self):
        hdc_net = yaw_height_hdc_network.yawHeightHdcNetwork()

        v_temp = self.view_templates[0]
        hdc_net.yaw_height_hdc_iteration(self.visual_odometer.delta, v_temp)
        # pcmax = pcnet.get_pc_max(pcnet.avg_xywrap, pcnet.avg_thwrap)

        return hdc_net

    def initialize_experience_map(self):
        emap = multilayered_experience_map.ExperienceMap()

        current_vt = self.view_templates.current_template

        emap.update(self.visual_odometer.delta[0],
                    self.visual_odometer.delta[1],
                    self.visual_odometer.delta[2],
                    self.grid_cell_network.MAX_ACTIVE_XYZ_PATH[0],
                    self.grid_cell_network.MAX_ACTIVE_XYZ_PATH[1],
                    self.grid_cell_network.MAX_ACTIVE_XYZ_PATH[2],
                    self.yaw_height_hdc_network.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH[0],
                    self.yaw_height_hdc_network.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH[1],
                    current_vt)

        return emap


    def current_exp(self):
        return self.experience_map.current_exp

    def evolve(self):
        c = self.time_step
        self.time_step += 1

        self.current_image = cv2.imread(self.image_source_list[c], cv2.IMREAD_GRAYSCALE)

        gc_max = self.grid_cell_network.get_gc_xyz()
        hdc_max = self.yaw_height_hdc_network.get_current_yaw_height_value()


        # get visual template
        v_temp = self.view_templates.match(self.current_image, gc_max[0], gc_max[1],
                                           gc_max[2], hdc_max[0], hdc_max[1])


        # get odometry
        self.current_odo = self.visual_odometer.update(self.current_image)

        # get gcmax hdcmax
        self.grid_cell_network.gc_iteration(self.visual_odometer.delta, v_temp)
        gc_max = self.grid_cell_network.get_gc_xyz()

        self.yaw_height_hdc_network.yaw_height_hdc_iteration(self.visual_odometer.delta, v_temp)
        hdc_max = self.yaw_height_hdc_network.get_current_yaw_height_value()

        self.current_grid_cell = gc_max
        self.current_hdc_cell = hdc_max

        self.experience_map.update(self.visual_odometer.delta[0],
                                   self.visual_odometer.delta[1],
                                   self.visual_odometer.delta[2],
                                   gc_max[0], gc_max[1], gc_max[2],
                                   hdc_max[0], hdc_max[1], v_temp)
