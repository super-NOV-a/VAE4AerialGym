import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import re
import traceback

# --- 1. 配置：设置您的文件路径 ---

# (必须) 设置您的 pareto_data.txt 文件路径
input_data_file = 'pareto_data.txt'
font_path = '/usr/share/fonts/MyFonts/simhei.ttf'  # 这是一个在Linux上常见的备用路径

# (可选) 设置输出图像的文件名
output_plot_file = 'pareto_front_comparison.png'


# --- 2. 解析 TXT 文件的函数 (已更新为包含beta) ---

def parse_pareto_txt_with_beta(filepath):
    """
    解析函数，读取 beta, kl, recon。
    返回一个字典，键是模型名称，值是 (KL, Recon, Beta) 元组的列表。
    """
    all_data = {}
    current_model = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 匹配模型名称
                if line.startswith('--- 模型: '):
                    match = re.search(r'--- 模型: (.*) ---', line)
                    if match:
                        current_model = match.group(1).strip()
                        all_data[current_model] = []
                        print(f"解析到模型: {current_model}")
                    else:
                        current_model = None

                # 匹配数据行
                elif current_model and line and not line.startswith(('#', 'beta,', '=')):
                    try:
                        parts = line.split(',')
                        if len(parts) == 3:
                            # 提取 beta, kl, recon
                            beta = float(parts[0].strip())
                            kl = float(parts[1].strip())
                            recon = float(parts[2].strip())

                            # 存储为 (X轴, Y轴, 标注)
                            all_data[current_model].append((kl, recon, beta))
                        else:
                            print(f"警告: 跳过格式不正确的数据行: {line}")
                    except ValueError:
                        print(f"警告: 无法解析行中的浮点数: {line}")

    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{filepath}'。请检查 input_data_file 路径。")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    return all_data


def plot_pareto_comparison_annotated_zh(model_data, save_path, font_prop=None):
    """
    绘制帕累托前沿对比图 (中文版，带beta标注)。
    """
    if not model_data:
        print("错误: 没有解析到任何模型数据。")
        return

    plt.figure(figsize=(12, 9))
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_data)))
    markers = ['o', 's', 'v', '^', 'D', '<', '>']
    has_data = False
    global_y_min, global_y_max = np.inf, -np.inf

    for i, (model_name, points) in enumerate(model_data.items()):
        valid_points = [(kl, recon, beta) for kl, recon, beta in points if kl > 0 and recon > 0]

        if not valid_points:
            print(f"警告: 模型 '{model_name}' 没有 KL > 0 且 Recon > 0 的数据点可绘制。")
            continue

        has_data = True

        np_points = np.array(valid_points)
        x_coords = np_points[:, 0]
        y_coords = np_points[:, 1]
        betas = np_points[:, 2]

        global_y_min = min(global_y_min, np.min(y_coords))
        global_y_max = max(global_y_max, np.max(y_coords))

        sorted_indices = np.argsort(x_coords)
        x_sorted = x_coords[sorted_indices]
        y_sorted = y_coords[sorted_indices]
        betas_sorted = betas[sorted_indices]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(x_sorted, y_sorted, marker=marker, linestyle='--',
                 label=model_name, markersize=12, linewidth=2.5, color=color)

        for k, r, b in zip(x_sorted, y_sorted, betas_sorted):
            beta_str = f"{b:g}"
            plt.annotate(f"β={beta_str}", (k, r),
                         textcoords="offset points",
                         xytext=(6, 6),
                         ha='left',
                         fontsize=12,
                         fontproperties=font_prop,
                         color=color)

    if not has_data:
        print("错误: 没有任何模型的数据可供绘制 (Log Scale 需要正值)。")
        plt.close()
        return

    # --- 修正字体设置 ---
    title = r'不同 $\beta$ 下 DC-VAE 与 β-VAE 的 $L_{KL}$-$L_{\mathrm{recon}}$ 对比'
    xlabel = 'KL 散度 ($L_{KL}$) - 潜在空间复杂度'
    ylabel = '重建损失 ($L_{\mathrm{recon}}$) - 任务精度'

    # 方法1：分别设置字体大小（推荐）
    # 在函数中添加以下代码替换原来的标题设置部分：

    # 创建字体字典
    title_font_dict = {'fontsize': 20}
    if font_prop:
        title_font_dict['fontproperties'] = font_prop

    plt.title(title, fontdict=title_font_dict)
    plt.xlabel(xlabel, fontsize=18, fontproperties=font_prop)
    plt.ylabel(ylabel, fontsize=18, fontproperties=font_prop)

    # 方法2：使用fontdict（如果需要设置多个字体属性）
    # font_dict = {'fontsize': 20, 'fontproperties': font_prop}
    # plt.title(title, fontdict=font_dict)
    # plt.xlabel(xlabel, fontdict={'fontsize': 18, 'fontproperties': font_prop})
    # plt.ylabel(ylabel, fontdict={'fontsize': 18, 'fontproperties': font_prop})

    # 图例设置
    if font_prop:
        # 如果有中文字体属性，确保字体大小也传递
        font_prop_with_size = font_prop.copy()
        font_prop_with_size.set_size(18)
        plt.legend(prop=font_prop_with_size, loc='best')
    else:
        plt.legend(loc='best', fontsize=18)

    # 刻度设置
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xscale('log')

    y_range = global_y_max - global_y_min
    y_padding = y_range * 0.25 if y_range > 1e-6 else max(abs(global_y_min) * 0.25, 1.5)
    plt.ylim(0, global_y_max + y_padding)

    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n帕累托对比图像 (Log Scale, 大字体) 已成功保存到: {save_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")
    finally:
        plt.close()  # 确保图形被关闭


# --- 4. 主执行流程 ---
if __name__ == "__main__":

    # --- 设置中文字体 ---
    font_prop = None
    if os.path.exists(font_path):
        try:
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            print(f"成功加载中文字体: {font_prop.get_name()} (from {font_path})")
        except Exception as e:
            print(f"加载字体 {font_path} 失败: {e}. 将使用默认字体。")
            traceback.print_exc()
    else:
        print(f"警告: 找不到中文字体文件 {font_path}。")
        print("请检查 font_path 变量是否指向了您系统上的一个有效 .ttf 或 .ttc 文件。")
        print("将使用默认字体，中文可能显示为方框。")

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # --- 运行解析和绘图 ---
    print(f"\n开始解析文件: {input_data_file}")
    all_data = parse_pareto_txt_with_beta(input_data_file)

    if all_data:
        print(f"成功解析到 {len(all_data)} 个模型的数据 (包含beta值)。")
        print("开始绘制带标注的中文对比图...")
        plot_pareto_comparison_annotated_zh(all_data, output_plot_file, font_prop=font_prop)
    else:
        print("解析数据失败，无法生成图像。")