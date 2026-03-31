import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os
import glob
from matplotlib import font_manager
# 手动指定字体路径
font_path = '/usr/share/fonts/MyFonts/simhei.ttf'  # 替换为实际路径
font_prop = font_manager.FontProperties(fname=font_path)

# 设置字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os
import glob
from scipy.ndimage import gaussian_filter1d  # 用于高斯平滑

# 1. 读取所有CSV文件
csv_files = glob.glob('*.csv')  # 获取当前目录所有CSV文件
if len(csv_files) < 6:
    print(f"找到 {len(csv_files)} 个CSV文件，但需要6个。请检查目录。")
    exit()

data_frames = {}
for file in csv_files[:6]:  # 取前6个文件
    df_name = os.path.splitext(os.path.basename(file))[0]  # 提取不带扩展名的文件名
    data_frames[df_name] = pd.read_csv(file)

# 2. 创建图形
plt.figure(figsize=(12, 8), dpi=100)

# 3. 对文件名进行排序
sorted_names = sorted(data_frames.keys())  # 按字母顺序排序文件名

# 4. 为每个文件绘制曲线和误差带
colors = plt.cm.tab10.colors  # 使用预定义颜色

for i, name in enumerate(sorted_names):
    df = data_frames[name]

    # 确保数据按step排序
    df = df.sort_values('Step')  # 注意列名大小写

    # 提取原始数据点
    steps = df['Step'].values
    values = df['Value'].values

    # 方法1：高斯平滑（保留端点）
    sigma = 2  # 平滑参数，值越大越平滑
    smoothed_values = gaussian_filter1d(values, sigma=sigma, mode='nearest')

    # 方法2：移动平均（保留端点）- 替代方案
    # window_size = 15
    # smoothed_values = values.copy()
    # for j in range(len(values)):
    #     start_idx = max(0, j - window_size//2)
    #     end_idx = min(len(values), j + window_size//2 + 1)
    #     smoothed_values[j] = np.mean(values[start_idx:end_idx])

    # 计算误差带 - 使用原始数据点与平滑值的差异
    # 保留端点计算
    errors = np.abs(values - smoothed_values)

    # 绘制原始数据点（小点）
    # plt.scatter(steps, values, s=15, color=colors[i], alpha=0.4, label='_nolegend_')

    # 绘制平滑曲线
    plt.plot(steps, smoothed_values, lw=1.5, label=name, color=colors[i])

    # 绘制误差带（保留端点）
    plt.fill_between(
        steps,
        smoothed_values - errors,
        smoothed_values + errors,
        color=colors[i],
        alpha=0.15
    )

    # 标记初始值和终止值
    # plt.scatter(steps[0], values[0], s=80, color=colors[i], marker='o', edgecolor='k', zorder=10, label='_nolegend_')
    # plt.scatter(steps[-1], values[-1], s=80, color=colors[i], marker='s', edgecolor='k', zorder=10, label='_nolegend_')

    # 添加初始值和终止值标签
    # plt.annotate(f'Start: {values[0]:.2f}',
    #              (steps[0], values[0]),
    #              xytext=(steps[0] - 10, values[0] + np.max(errors) * 0.5),
    #              arrowprops=dict(arrowstyle="->", color=colors[i], alpha=0.7),
    #              fontsize=9, color=colors[i])
    #
    # plt.annotate(f'End: {values[-1]:.2f}',
    #              (steps[-1], values[-1]),
    #              xytext=(steps[-1] + 10, values[-1] - np.max(errors) * 0.5),
    #              arrowprops=dict(arrowstyle="->", color=colors[i], alpha=0.7),
    #              fontsize=9, color=colors[i])


# 5. 添加图例和标签
plt.title('导航任务回报值', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Return', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(frameon=True)

# 6. 显示和保存
plt.tight_layout()
plt.savefig('combined_smooth_plot.png', bbox_inches='tight')
plt.show()