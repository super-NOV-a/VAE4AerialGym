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


def extract_latent_dim_from_filename(filename):
    """从文件名中提取潜在维度值"""
    patterns = [
        r'beta100\.0_LD_(\d+)',
        r'LD_(\d+)',
        r'latent_(\d+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))

    # 如果上述模式都不匹配，尝试从文件名中提取数字
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # 返回最大的数字（假设是潜在维度）
        return int(max(numbers, key=int))

    return None


def smooth_data(values, window_size=50):
    """使用移动平均平滑数据"""
    if len(values) < window_size:
        return values, list(range(len(values)))
    smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
    return smoothed, list(range(window_size - 1, len(values)))


def get_smoothed_final_value(values, window_size=50):
    """获取平滑后的最后一个值"""
    if len(values) < window_size:
        return values[-1] if values else float('nan')

    smoothed_values, _ = smooth_data(values, window_size)
    return smoothed_values[-1] if len(smoothed_values) > 0 else float('nan')


def read_tensorboard_logs_fixed_beta(log_dir_pattern):
    """读取固定beta=100.0，不同潜在维度的所有TensorBoard日志文件"""
    log_files = glob.glob(log_dir_pattern)
    print(f"找到 {len(log_files)} 个匹配的日志文件")

    all_data = {}

    for log_path in log_files:
        latent_dim = extract_latent_dim_from_filename(log_path)
        if latent_dim is None:
            print(f"无法从 {log_path} 提取潜在维度，跳过")
            continue

        print(f"处理 潜在维度={latent_dim} 的日志: {os.path.basename(log_path)}")

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
                all_data[latent_dim] = data
            else:
                print(f"  跳过 潜在维度={latent_dim} 的日志，因为缺少必要的数据")

        except Exception as e:
            print(f"  处理日志文件时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_data


def plot_latent_dim_comparison(all_data, beta=str(3.0)):
    """绘制不同潜在维度的对比图"""
    save_dir = "./beta"+beta+"_latent_dim_comparison_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置颜色和线型
    colors = plt.cm.plasma(np.linspace(0, 1, len(all_data)))
    line_styles = ['-', '--', '-.', ':'] * 5

    # 绘制KL散度对比图
    plt.figure(figsize=(12, 8))
    for i, (latent_dim, data) in enumerate(sorted(all_data.items())):
        color = colors[i]
        linestyle = line_styles[i % len(line_styles)]

        plt.plot(data['kl_steps'], data['kl_values'],
                 label=f'LD={latent_dim}', color=color, linestyle=linestyle, linewidth=2)

    plt.xlabel('训练步数 (Step)', fontsize=14, fontproperties=font_prop)
    plt.ylabel('KL散度 (Nats)', fontsize=14, fontproperties=font_prop)
    plt.title('β='+beta+'时不同潜在维度下的KL散度训练曲线对比', fontsize=16, fontproperties=font_prop)
    plt.legend(prop=font_prop, loc='best')
    plt.grid(True, alpha=0.3)

    # 如果数据范围很大，使用对数刻度
    kl_values = [value for data in all_data.values() for value in data['kl_values']]
    if max(kl_values) > 100:
        plt.yscale('log')

    plt.tight_layout()

    kl_save_path = os.path.join(save_dir, "kl_divergence_latent_dim_comparison.png")
    plt.savefig(kl_save_path, dpi=300, bbox_inches='tight')
    print(f"KL散度潜在维度对比图已保存: {kl_save_path}")
    plt.show()

    # 绘制重建MSE对比图
    plt.figure(figsize=(12, 8))
    for i, (latent_dim, data) in enumerate(sorted(all_data.items())):
        color = colors[i]
        linestyle = line_styles[i % len(line_styles)]

        plt.plot(data['mse_steps'], data['mse_values'],
                 label=f'LD={latent_dim}', color=color, linestyle=linestyle, linewidth=2)

    plt.xlabel('训练步数 (Step)', fontsize=14, fontproperties=font_prop)
    plt.ylabel('重建MSE', fontsize=14, fontproperties=font_prop)
    plt.title('β='+beta+'时不同潜在维度下的重建MSE训练曲线对比', fontsize=16, fontproperties=font_prop)
    plt.legend(prop=font_prop, loc='best')
    plt.grid(True, alpha=0.3)

    # 如果数据范围很大，使用对数刻度
    mse_values = [value for data in all_data.values() for value in data['mse_values']]
    if max(mse_values) > 100:
        plt.yscale('log')

    plt.tight_layout()

    mse_save_path = os.path.join(save_dir, "reconstruction_mse_latent_dim_comparison.png")
    plt.savefig(mse_save_path, dpi=300, bbox_inches='tight')
    print(f"重建MSE潜在维度对比图已保存: {mse_save_path}")
    plt.show()


def plot_latent_dim_smoothed_comparison(all_data, window_size=50, beta=str(3.0)):
    save_dir = "./beta"+beta+"_latent_dim_comparison_plots"
    """绘制平滑后的潜在维度对比图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置颜色和线型
    colors = plt.cm.plasma(np.linspace(0, 1, len(all_data)))
    line_styles = ['-', '--', '-.', ':'] * 5

    # 绘制平滑后的KL散度对比图
    plt.figure(figsize=(12, 8))
    for i, (latent_dim, data) in enumerate(sorted(all_data.items())):
        color = colors[i]
        linestyle = line_styles[i % len(line_styles)]

        if len(data['kl_values']) >= window_size:
            smoothed_kl, smoothed_steps = smooth_data(data['kl_values'], window_size)
            # 确保步数对应正确
            actual_steps = [data['kl_steps'][i] for i in smoothed_steps]
            plt.plot(actual_steps, smoothed_kl,
                     label=f'LD={latent_dim}', color=color, linestyle=linestyle, linewidth=2)

    plt.xlabel('训练步数 (Step)', fontsize=14, fontproperties=font_prop)
    plt.ylabel('KL散度 (Nats)', fontsize=14, fontproperties=font_prop)
    plt.title('β='+beta+'时不同潜在维度下的KL散度训练曲线对比（平滑）', fontsize=16, fontproperties=font_prop)
    plt.legend(prop=font_prop, loc='best')
    plt.grid(True, alpha=0.3)

    # 如果数据范围很大，使用对数刻度
    kl_values = [value for data in all_data.values() for value in data['kl_values']]
    if max(kl_values) > 100:
        plt.yscale('log')

    plt.tight_layout()

    kl_save_path = os.path.join(save_dir, "kl_divergence_latent_dim_smoothed_comparison.png")
    plt.savefig(kl_save_path, dpi=300, bbox_inches='tight')
    print(f"平滑KL散度潜在维度对比图已保存: {kl_save_path}")
    plt.show()

    # 绘制平滑后的重建MSE对比图
    plt.figure(figsize=(12, 8))
    for i, (latent_dim, data) in enumerate(sorted(all_data.items())):
        color = colors[i]
        linestyle = line_styles[i % len(line_styles)]

        if len(data['mse_values']) >= window_size:
            smoothed_mse, smoothed_steps = smooth_data(data['mse_values'], window_size)
            # 确保步数对应正确
            actual_steps = [data['mse_steps'][i] for i in smoothed_steps]
            plt.plot(actual_steps, smoothed_mse,
                     label=f'LD={latent_dim}', color=color, linestyle=linestyle, linewidth=2)

    plt.xlabel('训练步数 (Step)', fontsize=14, fontproperties=font_prop)
    plt.ylabel('重建MSE', fontsize=14, fontproperties=font_prop)
    plt.title('β='+beta+'时不同潜在维度下的重建MSE训练曲线对比（平滑）', fontsize=16, fontproperties=font_prop)
    plt.legend(prop=font_prop, loc='best')
    plt.grid(True, alpha=0.3)

    # 如果数据范围很大，使用对数刻度
    mse_values = [value for data in all_data.values() for value in data['mse_values']]
    if max(mse_values) > 100:
        plt.yscale('log')

    plt.tight_layout()

    mse_save_path = os.path.join(save_dir, "reconstruction_mse_latent_dim_smoothed_comparison.png")
    plt.savefig(mse_save_path, dpi=300, bbox_inches='tight')
    print(f"平滑重建MSE潜在维度对比图已保存: {mse_save_path}")
    plt.show()


def generate_latent_dim_summary_table(all_data, window_size=50, beta=str(3.0)):
    """生成潜在维度数据汇总表格，使用平滑后的最终值"""
    save_dir = "./beta"+beta+"_latent_dim_comparison_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n" + "=" * 70)
    print("潜在维度训练数据汇总 (β="+beta+")")
    print("=" * 70)
    print(f"{'潜在维度':<12} | {'KL数据点':<12} | {'MSE数据点':<12} | {'最终KL':<15} | {'最终MSE':<15}")
    print("-" * 70)

    summary_data = []
    for latent_dim, data in sorted(all_data.items()):
        kl_points = len(data['kl_steps'])
        mse_points = len(data['mse_steps'])

        # 使用平滑后的最终值
        final_kl = get_smoothed_final_value(data['kl_values'], window_size)
        final_mse = get_smoothed_final_value(data['mse_values'], window_size)

        print(f"{latent_dim:<12} | {kl_points:<12} | {mse_points:<12} | {final_kl:<15.6f} | {final_mse:<15.6f}")
        summary_data.append([latent_dim, kl_points, mse_points, final_kl, final_mse])

    print("=" * 70)

    # 保存汇总表格到文件
    summary_path = os.path.join(save_dir, "latent_dim_training_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("潜在维度训练数据汇总 (β="+beta+")\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'潜在维度':<12} | {'KL数据点':<12} | {'MSE数据点':<12} | {'最终KL':<15} | {'最终MSE':<15}\n")
        f.write("-" * 70 + "\n")
        for row in summary_data:
            f.write(f"{row[0]:<12} | {row[1]:<12} | {row[2]:<12} | {row[3]:<15.6f} | {row[4]:<15.6f}\n")
        f.write("=" * 70 + "\n")

    print(f"潜在维度汇总表格已保存: {summary_path}")

    return summary_data


def plot_convergence_comparison(all_data, window_size=50, beta=str(3.0)):
    """绘制收敛性能对比图，使用平滑后的最终值"""
    save_dir = "./beta"+beta+"_latent_dim_comparison_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 提取平滑后的最终性能指标
    latent_dims = sorted(all_data.keys())
    final_kl = [get_smoothed_final_value(all_data[ld]['kl_values'], window_size) for ld in latent_dims]
    final_mse = [get_smoothed_final_value(all_data[ld]['mse_values'], window_size) for ld in latent_dims]

    # 绘制最终性能对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 最终KL散度对比
    ax1.plot(latent_dims, final_kl, 'o-', linewidth=2, markersize=8, label='最终KL散度')
    ax1.set_xlabel('潜在维度', fontsize=14, fontproperties=font_prop)
    ax1.set_ylabel('KL散度 (Nats)', fontsize=14, fontproperties=font_prop)
    ax1.set_title('β='+beta+'时不同潜在维度的最终KL散度', fontsize=16, fontproperties=font_prop)
    ax1.grid(True, alpha=0.3)

    # 最终MSE对比
    ax2.plot(latent_dims, final_mse, 'o-', linewidth=2, markersize=8, label='最终重建MSE')
    ax2.set_xlabel('潜在维度', fontsize=14, fontproperties=font_prop)
    ax2.set_ylabel('重建MSE', fontsize=14, fontproperties=font_prop)
    ax2.set_title('β='+beta+'时不同潜在维度的最终重建MSE', fontsize=16, fontproperties=font_prop)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    convergence_save_path = os.path.join(save_dir, "latent_dim_convergence_comparison.png")
    plt.savefig(convergence_save_path, dpi=300, bbox_inches='tight')
    print(f"潜在维度收敛性能对比图已保存: {convergence_save_path}")
    plt.show()


if __name__ == "__main__":
    # 配置参数 - 针对固定beta=100.0，变化潜在维度
    beta = str(3.0)
    log_patterns = [
        # "./runs/dc_vae_betaScaled_exact_beta100.0_LD_*/events.out.tfevents.*",
        # "./runs/dc_vae_betaScaled_exact_beta100.0_LD_*/*/events.out.tfevents.*",
        "/home/niu/workspaces/VAE_ws/agent_encoder/runs/dc_vae_beta"+beta+"_LD_*/events.out.tfevents.*",
        # 添加更多可能的模式...
    ]

    all_data = {}
    for pattern in log_patterns:
        print(f"尝试模式: {pattern}")
        data = read_tensorboard_logs_fixed_beta(pattern)
        if data:
            all_data.update(data)
            print(f"从该模式找到 {len(data)} 个日志文件")
        else:
            print("该模式未找到匹配的日志文件")

    if not all_data:
        print("未找到任何匹配的日志文件，请检查日志模式设置")
        exit(1)

    # 设置平滑窗口大小
    window_size = 50

    # 生成汇总表格（使用平滑后的最终值）
    summary_data = generate_latent_dim_summary_table(all_data, window_size, beta)

    # 绘制对比图
    plot_latent_dim_comparison(all_data, beta)

    # 绘制平滑后的对比图
    plot_latent_dim_smoothed_comparison(all_data, window_size, beta)

    # 绘制收敛性能对比图（使用平滑后的最终值）
    plot_convergence_comparison(all_data, window_size, beta)

    print("\n所有潜在维度对比图表生成完成！")