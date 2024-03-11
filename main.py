# -*- coding: utf-8 -*-
# @Time : 2023/3/10 22:00
# @Author : xiao cong
# @Description :

import neuroSLAM

import matplotlib.pyplot as plt
import os
import numpy as np


def load_image(image_path):
    img_list = []
    for img_name in os.listdir(image_path):
        img_list.append(image_path + img_name)
    return img_list


def main():
    image_path = "D:\\Works\Datasets\\2023-05-14-22-38-19\\mav0\\cam0\\data\\"  # opencv 读取不能有中文路径
    image_source_list = load_image(image_path)
    driver = neuroSLAM.neuroSLAM(image_source_list)
    odos = []
    exps = []

    n_steps = 1000
    ########################################################################
    for i in range(n_steps):
        print("正在处理第" + str(i + 1) + "帧图像......")
        # do a time step of the simulation
        driver.evolve()

        # query some values for plotting
        im = driver.current_image
        emap = driver.experience_map
        gc_max = driver.current_grid_cell
        hdc_max = driver.current_hdc_cell

        odo = driver.current_odo
        odos.append([odo[0], odo[1], odo[2]])

        # current_exp = driver.current_exp
        # exp = [current_exp.x_exp, current_exp.y_exp, current_exp.z_exp]
        # exps.append(exp)

    for i in range(len(emap.exps)):
        exp = [emap.exps[i].x_exp, emap.exps[i].y_exp, emap.exps[i].z_exp]
        exps.append(exp)

    fig = plt.figure(figsize=(14, 14), dpi=100)
    ax1 = fig.add_subplot(121, projection='3d')
    odos = np.array(odos)
    np.savetxt("./datasets/odo_map.txt", odos)
    x1 = odos[:, 0]
    y1 = odos[:, 1]
    z1 = odos[:, 2]
    ax1.scatter(x1, y1, z1)
    plt.title('Odometry')

    ax2 = fig.add_subplot(122, projection='3d')
    exps = np.array(exps)
    np.savetxt("./datasets/exp_map.txt", exps)
    x2 = exps[:, 0]
    y2 = exps[:, 1]
    z2 = exps[:, 2]
    ax2.scatter(x2, y2, z2)
    plt.title('Experience Map')
    plt.show()


if __name__ == "__main__":
    main()
