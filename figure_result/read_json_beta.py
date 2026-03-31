import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.font_manager as fm

# 设置中文字体
font_path = '/usr/share/fonts/MyFonts/simhei.ttf'  # 替换为你的中文字体路径
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False


def extract_beta_from_filename(filename):
    """从文件名中提取beta值"""
    # 尝试匹配多种可能的beta值格式
    patterns = [
        r'betaScaled_exact_beta([\d.]+)_LD_64',
        r'beta_([\d.]+)_LD_64',
        r'b([\d.]+)_LD_64'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return float(match.group(1))

    # 如果上述模式都不匹配，尝试从文件名中提取数字
    numbers = re.findall(r'\d+\.?\d*', filename)
    if numbers:
        # 返回最大的数字（假设是beta值）
        return float(max(numbers, key=float))

    return None


def read_tensorboard_logs(log_dir_pattern):
    """读取匹配模式的所有TensorBoard日志文件"""
    log_files = glob.glob(log_dir_pattern)
    print(f"找到 {len(log_files)} 个匹配的日志文件")

    all_data = {}

    for log_path in log_files:
        beta = extract_beta_from_filename(log_path)
        if beta is None:
            print(f"无法从 {log_path} 提取beta值，跳过")
            continue

        print(f"处理 beta={beta} 的日志: {os.path.basename(log_path)}")

        try:
            # 创建事件累积器
            event_acc = EventAccumulator(log_path)
            event_acc.Reload()

            # 获取所有标量标签
            tags = event_acc.Tags()['scalars']
            print(f"  可用标签: {tags}")

            # 读取我们需要的数据
            data = {}

            # 尝试不同的标签名称变体
            kl_tags = ['step_kl_nats_unweighted', 'Loss/step_kl_nats_unweighted',
                       'kl_nats_unweighted', 'Loss/kl_nats_unweighted',
                       'kl_loss', 'Loss/kl_loss']

            kl_tag_found = None
            for tag in kl_tags:
                if tag in tags:
                    kl_tag_found = tag
                    break

            if kl_tag_found:
                kl_events = event_acc.Scalars(kl_tag_found)
                data['kl_steps'] = [e.step for e in kl_events]
                data['kl_values'] = [e.value for e in kl_events]
                print(f"  KL数据点 ({kl_tag_found}): {len(data['kl_steps'])}")
            else:
                print(f"  警告: 未找到 KL散度 标签，尝试的标签: {kl_tags}")
                # 不立即跳过，可能还有其他数据
                data['kl_steps'] = []
                data['kl_values'] = []

            # 尝试不同的MSE标签名称变体
            mse_tags = ['step_recon_mse_unweighted', 'Loss/step_recon_mse_unweighted',
                        'recon_mse_unweighted', 'Loss/recon_mse_unweighted',
                        'recon_loss', 'Loss/recon_loss', 'mse_loss', 'Loss/mse_loss']

            mse_tag_found = None
            for tag in mse_tags:
                if tag in tags:
                    mse_tag_found = tag
                    break

            if mse_tag_found:
                mse_events = event_acc.Scalars(mse_tag_found)
                data['mse_steps'] = [e.step for e in mse_events]
                data['mse_values'] = [e.value for e in mse_events]
                print(f"  MSE数据点 ({mse_tag_found}): {len(data['mse_steps'])}")
            else:
                print(f"  警告: 未找到 重建MSE 标签，尝试的标签: {mse_tags}")
                data['mse_steps'] = []
                data['mse_values'] = []

            # 只有当两个指标都有数据时才添加到结果中
            if data['kl_steps'] and data['mse_steps']:
                all_data[beta] = data
            else:
                print(f"  跳过 beta={beta} 的日志，因为缺少必要的数据")

        except Exception as e:
            print(f"  处理日志文件时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_data


def plot_comparison(all_data, save_dir="./beta_comparison_plots"):
    """绘制对比图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置颜色和线型
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))
    line_styles = ['-', '--', '-.', ':'] * 5

    # 绘制KL散度对比图
    plt.figure(figsize=(12, 8))
    for i, (beta, data) in enumerate(sorted(all_data.items())):
        color = colors[i]
        linestyle = line_styles[i % len(line_styles)]

        # 对beta=1000.0使用不同的标记，使其在图中更明显
        if beta == 1000.0:
            plt.plot(data['kl_steps'], data['kl_values'],
                     label=f'β={beta}', color=color, linestyle=linestyle,
                     linewidth=3, marker='o', markersize=2, markevery=50)
        else:
            plt.plot(data['kl_steps'], data['kl_values'],
                     label=f'β={beta}', color=color, linestyle=linestyle, linewidth=2)

    plt.xlabel('训练步数 (Step)', fontsize=14, fontproperties=font_prop)
    plt.ylabel('KL散度 (Nats)', fontsize=14, fontproperties=font_prop)
    plt.title('不同β值下的KL散度训练曲线对比', fontsize=16, fontproperties=font_prop)
    plt.legend(prop=font_prop, loc='best')
    plt.grid(True, alpha=0.3)

    # 如果数据范围很大，使用对数刻度
    if max(max(data['kl_values']) for data in all_data.values()) > 100:
        plt.yscale('log')

    plt.tight_layout()

    kl_save_path = os.path.join(save_dir, "kl_divergence_comparison.png")
    plt.savefig(kl_save_path, dpi=300, bbox_inches='tight')
    print(f"KL散度对比图已保存: {kl_save_path}")
    plt.show()

    # 绘制重建MSE对比图
    plt.figure(figsize=(12, 8))
    for i, (beta, data) in enumerate(sorted(all_data.items())):
        color = colors[i]
        linestyle = line_styles[i % len(line_styles)]

        # 对beta=1000.0使用不同的标记
        if beta == 1000.0:
            plt.plot(data['mse_steps'], data['mse_values'],
                     label=f'β={beta}', color=color, linestyle=linestyle,
                     linewidth=3, marker='o', markersize=2, markevery=50)
        else:
            plt.plot(data['mse_steps'], data['mse_values'],
                     label=f'β={beta}', color=color, linestyle=linestyle, linewidth=2)

    plt.xlabel('训练步数 (Step)', fontsize=14, fontproperties=font_prop)
    plt.ylabel('重建MSE', fontsize=14, fontproperties=font_prop)
    plt.title('不同β值下的重建MSE训练曲线对比', fontsize=16, fontproperties=font_prop)
    plt.legend(prop=font_prop, loc='best')
    plt.grid(True, alpha=0.3)

    # 如果数据范围很大，使用对数刻度
    if max(max(data['mse_values']) for data in all_data.values()) > 100:
        plt.yscale('log')

    plt.tight_layout()

    mse_save_path = os.path.join(save_dir, "reconstruction_mse_comparison.png")
    plt.savefig(mse_save_path, dpi=300, bbox_inches='tight')
    print(f"重建MSE对比图已保存: {mse_save_path}")
    plt.show()


def plot_smoothed_comparison(all_data, window_size=50, save_dir="./beta_comparison_plots"):
    """绘制平滑后的对比图（可选）"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def smooth_data(values, window_size):
        """使用移动平均平滑数据"""
        if len(values) < window_size:
            return values, list(range(len(values)))
        smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        return smoothed, list(range(window_size - 1, len(values)))

    # 设置颜色和线型
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))
    line_styles = ['-', '--', '-.', ':'] * 5

    # 绘制平滑后的KL散度对比图
    plt.figure(figsize=(12, 8))
    for i, (beta, data) in enumerate(sorted(all_data.items())):
        color = colors[i]
        linestyle = line_styles[i % len(line_styles)]

        if len(data['kl_values']) >= window_size:
            smoothed_kl, smoothed_steps = smooth_data(data['kl_values'], window_size)
            # 确保步数对应正确
            actual_steps = [data['kl_steps'][i] for i in smoothed_steps]
            plt.plot(actual_steps, smoothed_kl,
                     label=f'β={beta}', color=color, linestyle=linestyle, linewidth=2)

    plt.xlabel('训练步数 (Step)', fontsize=14, fontproperties=font_prop)
    plt.ylabel('KL散度 (Nats)', fontsize=14, fontproperties=font_prop)
    plt.title('不同β值下的KL散度训练曲线对比（平滑）', fontsize=16, fontproperties=font_prop)
    plt.legend(prop=font_prop, loc='best')
    plt.grid(True, alpha=0.3)

    # 如果数据范围很大，使用对数刻度
    if max(max(data['kl_values']) for data in all_data.values()) > 100:
        plt.yscale('log')

    plt.tight_layout()

    kl_save_path = os.path.join(save_dir, "kl_divergence_smoothed_comparison.png")
    plt.savefig(kl_save_path, dpi=300, bbox_inches='tight')
    print(f"平滑KL散度对比图已保存: {kl_save_path}")
    plt.show()

    # 绘制平滑后的重建MSE对比图
    plt.figure(figsize=(12, 8))
    for i, (beta, data) in enumerate(sorted(all_data.items())):
        color = colors[i]
        linestyle = line_styles[i % len(line_styles)]

        if len(data['mse_values']) >= window_size:
            smoothed_mse, smoothed_steps = smooth_data(data['mse_values'], window_size)
            # 确保步数对应正确
            actual_steps = [data['mse_steps'][i] for i in smoothed_steps]
            plt.plot(actual_steps, smoothed_mse,
                     label=f'β={beta}', color=color, linestyle=linestyle, linewidth=2)

    plt.xlabel('训练步数 (Step)', fontsize=14, fontproperties=font_prop)
    plt.ylabel('重建MSE', fontsize=14, fontproperties=font_prop)
    plt.title('不同β值下的重建MSE训练曲线对比（平滑）', fontsize=16, fontproperties=font_prop)
    plt.legend(prop=font_prop, loc='best')
    plt.grid(True, alpha=0.3)

    # 如果数据范围很大，使用对数刻度
    if max(max(data['mse_values']) for data in all_data.values()) > 100:
        plt.yscale('log')

    plt.tight_layout()

    mse_save_path = os.path.join(save_dir, "reconstruction_mse_smoothed_comparison.png")
    plt.savefig(mse_save_path, dpi=300, bbox_inches='tight')
    print(f"平滑重建MSE对比图已保存: {mse_save_path}")
    plt.show()


def generate_summary_table(all_data, save_dir="./beta_comparison_plots"):
    """生成数据汇总表格"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n" + "=" * 80)
    print("训练数据汇总")
    print("=" * 80)
    print(
        f"{'β值':<10} | {'KL数据点':<12} | {'MSE数据点':<12} | {'最终KL':<15} | {'最终MSE':<15} | {'最小KL':<15} | {'最小MSE':<15}")
    print("-" * 80)

    summary_data = []
    for beta, data in sorted(all_data.items()):
        kl_points = len(data['kl_steps'])
        mse_points = len(data['mse_steps'])
        final_kl = data['kl_values'][-1] if kl_points > 0 else float('nan')
        final_mse = data['mse_values'][-1] if mse_points > 0 else float('nan')
        min_kl = min(data['kl_values']) if kl_points > 0 else float('nan')
        min_mse = min(data['mse_values']) if mse_points > 0 else float('nan')

        print(
            f"{beta:<10} | {kl_points:<12} | {mse_points:<12} | {final_kl:<15.6f} | {final_mse:<15.6f} | {min_kl:<15.6f} | {min_mse:<15.6f}")
        summary_data.append([beta, kl_points, mse_points, final_kl, final_mse, min_kl, min_mse])

    print("=" * 80)

    # 保存汇总表格到文件
    summary_path = os.path.join(save_dir, "training_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("训练数据汇总\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'β值':<10} | {'KL数据点':<12} | {'MSE数据点':<12} | {'最终KL':<15} | {'最终MSE':<15} | {'最小KL':<15} | {'最小MSE':<15}\n")
        f.write("-" * 80 + "\n")
        for row in summary_data:
            f.write(
                f"{row[0]:<10} | {row[1]:<12} | {row[2]:<12} | {row[3]:<15.6f} | {row[4]:<15.6f} | {row[5]:<15.6f} | {row[6]:<15.6f}\n")
        f.write("=" * 80 + "\n")

    print(f"汇总表格已保存: {summary_path}")


if __name__ == "__main__":
    # 配置参数
    # 请根据你的实际目录结构修改这个模式
    # 尝试多种可能的目录结构
    log_patterns = [
        "/home/niu/workspaces/VAE_ws/agent_encoder/runs/dc_vae_beta*_LD_64*/events.out.tfevents.*",
        # "./runs/dc_vae_betaScaled_exact_beta*_LD_64*/*/events.out.tfevents.*",
        # "/home/niu/workspaces/VAE_ws/tensorboard_logs/dc_vae_betaScaled_exact_beta*_LD_64*/events.out.tfevents.*",
        # 添加更多可能的模式...
    ]

    all_data = {}
    for pattern in log_patterns:
        print(f"尝试模式: {pattern}")
        data = read_tensorboard_logs(pattern)
        if data:
            all_data.update(data)
            print(f"从该模式找到 {len(data)} 个日志文件")
        else:
            print("该模式未找到匹配的日志文件")

    if not all_data:
        print("未找到任何匹配的日志文件，请检查日志模式设置")
        # 尝试手动指定一些已知的beta值路径
        known_paths = [
            # 添加你知道的特定路径
        ]
        for path in known_paths:
            if os.path.exists(path):
                print(f"尝试已知路径: {path}")
                data = read_tensorboard_logs(path)
                if data:
                    all_data.update(data)

    if not all_data:
        print("仍然未找到任何日志文件，请手动检查路径")
        exit(1)

    # 生成汇总表格
    generate_summary_table(all_data)

    # 绘制对比图
    plot_comparison(all_data)

    # 可选：绘制平滑后的对比图
    plot_smoothed_comparison(all_data, window_size=50)

    print("\n所有图表生成完成！")