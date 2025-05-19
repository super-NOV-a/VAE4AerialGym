import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_multiple_csvs(csv_files, output_dir='output_plots'):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 遍历每个CSV文件
    for file in csv_files:
        # 读取CSV文件
        data = pd.read_csv(file)

        # 提取x和y数据
        x = data['Epoch']
        y1 = data['Average Total Loss']
        y2 = data['Average KL Loss']
        y3 = y1-y2

        # 获取文件名（不包含路径）
        file_name = os.path.basename(file)

        # 绘制每个y值
        ax.plot(x, y1, label=f'Total Loss ({file_name})', linestyle='-')
        ax.plot(x, y2, label=f'KL Loss ({file_name})', linestyle='--')
        # ax.plot(x, y3, label=f'Recon Loss ({file_name})', linestyle=':')

    # 设置标题和标签
    ax.set_title('Training Loss Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # 添加图例
    ax.legend()

    # 保存图像
    output_path = os.path.join(output_dir, 'combined_loss_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    # 显示图像
    plt.show()

if __name__ == "__main__":
    # CSV文件列表
    csv_files = [
        # "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/100_beta_10_LD_64.csv",
        # "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/101_beta_3_LD_64.csv",
        # "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/102_beta_1_LD_64.csv",
        # "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/110_beta_10_LD_64.csv",
        # "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/111_beta_3_LD_64.csv",
        # "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/112_beta_1_LD_64.csv",
        "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/611_beta_10_LD_64.csv",
        "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/611_beta_3_LD_64.csv",
        "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/agent_encoder/train_loss/611_beta_1_LD_64.csv",
    ]

    # 调用函数
    plot_multiple_csvs(csv_files)