import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_multiple_csvs(csv_files, output_dir='output_plots'):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建包含三个子图的画布 (3行1列)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # 设置子图标题
    titles = [
        'Average Total Loss Comparison',
        'Average KL Loss Comparison',
        'Reconstructed Loss (Total - beta*KL) Comparison'
    ]

    # 遍历每个CSV文件
    for file in csv_files:
        # 读取CSV文件
        data = pd.read_csv(file)

        # 解析文件名
        filename = os.path.basename(file).split('.')[0]  # 移除扩展名
        parts = filename.split('_')
        exp_id = parts[0]  # 实验编号 (如520)
        beta = parts[2]  # beta值 (如1/3/10)
        ld_dim = parts[-1]  # 潜在空间维度 (如64)

        # 计算重构损失
        total_loss = data['Average Total Loss']
        kl_loss = data['Average KL Loss']
        recon_loss = total_loss - float(beta)*64/(480*270) * kl_loss

        # 生成图例标签
        label = f"Exp{exp_id} (β={beta}, LD={ld_dim})"

        # 在三个子图中分别绘制曲线
        axes[0].plot(data['Epoch'], total_loss, label=label)
        axes[1].plot(data['Epoch'], kl_loss, label=label)
        axes[2].plot(data['Epoch'], recon_loss, label=label)

    # 统一设置子图格式
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss Value', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8, loc='upper right')

    # 调整布局
    plt.tight_layout(pad=3.0)

    # 保存图像
    output_path = os.path.join(output_dir, 'combined_loss_analysis.png')
    plt.savefig(output_path, dpi=300)
    print(f"Analysis plot saved to: {output_path}")

    # 显示图像
    plt.show()


if __name__ == "__main__":
    # CSV文件列表（示例路径）
    csv_files = [
        # "/home/niu/workspaces/VAE_ws/agent_encoder/train_loss/520_beta_1_LD_64.csv",
        "/home/niu/workspaces/VAE_ws/agent_encoder/train_loss/520_beta_3_LD_64.csv",
        "/home/niu/workspaces/VAE_ws/agent_encoder/train_loss/520_beta_10_LD_64.csv",
        # "/home/niu/workspaces/VAE_ws/agent_encoder/train_loss/521_beta_1_LD_64.csv",
        "/home/niu/workspaces/VAE_ws/agent_encoder/train_loss/521_beta_3_LD_64.csv",
        "/home/niu/workspaces/VAE_ws/agent_encoder/train_loss/521_beta_10_LD_64.csv",
    ]

    # 调用绘图函数
    plot_multiple_csvs(csv_files)