# -*- coding: utf-8 -*-
# @Time : 2023/3/14 16:42
# @Author : xiao cong
# @Description :

import numpy as np
import matplotlib.pyplot as plt

exps = np.loadtxt("exp_map.txt")
odos = np.loadtxt("odo_map.txt")


fig = plt.figure(figsize=(14, 14), dpi=100)
ax1 = fig.add_subplot(121, projection='3d')
x1 = np.round(odos[:, 0], 2)
y1 = np.round(odos[:, 1], 2)
z1 = np.round(odos[:, 2], 2)
ax1.plot(x1, y1, z1)
plt.title('Odometry')

ax2 = fig.add_subplot(122, projection='3d')
x2 = np.round(exps[:, 0], 2)
y2 = np.round(exps[:, 1], 2)
z2 = np.round(exps[:, 2], 2)
ax2.plot(x2, y2, z2)
plt.title('Experience Map')
plt.show()