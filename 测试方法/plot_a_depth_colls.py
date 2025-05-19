# 分别读取深度图和碰撞图，将其plot绘制在一起

import cv2
import numpy as np
import matplotlib.pyplot as plt


def custom_normalize(normalized_depth_map):
    """
    将深度图归一化到[0, 255]范围，(大于0)&(小于MIN_DEPTH)的像素值设置为MIN_DEPTH
    返回值范围[0, 255]的深度图
    """
    max_val = np.max(normalized_depth_map)
    if max_val > 255:
        normalized_depth_map = normalized_depth_map * 255 / 7000  # DIML/CVl RGB-D 除以1000得到米  不再使用 SUN RGB-D
    normalized_depth_map[(normalized_depth_map < 5) & (normalized_depth_map > 0)] = 5
    normalized_depth_map = cv2.resize(normalized_depth_map, (480, 270), interpolation=cv2.INTER_LINEAR)
    return normalized_depth_map.astype(np.uint8)

# 分别读取生成好的 深度图和碰撞图 ，将其plot绘制在一起

depth = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_15339.png"
coll = "/home/niu/workspaces/VAE_ws/datasets/colls_offset/depth_15339.png"
depth_image = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
coll_image = cv2.imread(coll, cv2.IMREAD_UNCHANGED)
depth_image = depth_image.astype(np.float32)
coll_image = coll_image.astype(np.float32)
depth_image = custom_normalize(depth_image)
coll_image = coll_image
depth_image = depth_image.squeeze()
coll_image = coll_image.squeeze()
plt.subplot(1, 2, 1)
plt.imshow(depth_image)
plt.title("Depth Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(coll_image)
plt.title("Collision Image")
plt.axis("off")
plt.tight_layout()
plt.show()
