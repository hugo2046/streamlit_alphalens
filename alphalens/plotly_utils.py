'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-14 08:56:31
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-20 13:37:51
FilePath: 
Description: 
'''
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def random_color():
    # 生成随机颜色代码
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    )

def get_rgb_color(index: int, total: int, cump: str = "coolwarm") -> str:
    norm = mcolors.Normalize(vmin=0, vmax=total - 1)
    cmap = cm.ScalarMappable(norm=norm, cmap=cump)  # cm.coolwarm
    color = cmap.to_rgba(index)[:3]  # 获取 RGB 颜色
    color = "rgb(" + ",".join([str(int(255 * c)) for c in color]) + ")"

    return color