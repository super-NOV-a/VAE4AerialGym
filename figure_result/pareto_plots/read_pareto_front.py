import os
import re
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import traceback


# --- beta 提取函数 (与之前相同) ---
def extract_beta_from_filename(filename):
    """从文件名中提取beta值"""
    patterns = [
        r'betaScaled_exact_beta([\d.]+)_LD_64',
        r'beta_([\d.]+)_LD_64',
        r'b([\d.]+)_LD_64'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return float(match.group(1))

    beta_match = re.search(r'beta([\d.]+)', filename, re.IGNORECASE)
    if beta_match:
        return float(beta_match.group(1))

    numbers = re.findall(r'\d+\.?\d*', os.path.basename(filename))
    if numbers:
        return float(max(numbers, key=float))

    print(f"警告: 无法从 {filename} 提取 beta 值")
    return None


# --- 日志读取函数 (与之前相同) ---
def read_tensorboard_logs(log_dir_pattern):
    """读取匹配模式的所有TensorBoard日志文件"""
    log_files = glob.glob(log_dir_pattern, recursive=True)
    print(f"模式 '{log_dir_pattern}' 找到 {len(log_files)} 个匹配的日志文件")

    all_data = {}
    for log_path in log_files:
        beta = extract_beta_from_filename(log_path)
        if beta is None:
            print(f"无法从 {log_path} 提取beta值，跳过")
            continue

        print(f"处理 beta={beta} 的日志: {os.path.basename(log_path)}")
        try:
            event_acc = EventAccumulator(log_path)
            event_acc.Reload()
            tags = event_acc.Tags()['scalars']
            data = {}

            kl_tags = ['step_kl_nats_unweighted', 'Loss/step_kl_nats_unweighted',
                       'kl_nats_unweighted', 'Loss/kl_nats_unweighted',
                       'kl_loss', 'Loss/kl_loss']
            kl_tag_found = next((tag for tag in kl_tags if tag in tags), None)

            if kl_tag_found:
                data['kl_values'] = [e.value for e in event_acc.Scalars(kl_tag_found)]
            else:
                print(f"  警告: 未找到 KL散度 标签")
                data['kl_values'] = []

            mse_tags = ['step_recon_mse_unweighted', 'Loss/step_recon_mse_unweighted',
                        'recon_mse_unweighted', 'Loss/recon_mse_unweighted',
                        'recon_loss', 'Loss/recon_loss', 'mse_loss', 'Loss/mse_loss']
            mse_tag_found = next((tag for tag in mse_tags if tag in tags), None)

            if mse_tag_found:
                data['mse_values'] = [e.value for e in event_acc.Scalars(mse_tag_found)]
            else:
                print(f"  警告: 未找到 重建MSE 标签")
                data['mse_values'] = []

            if data['kl_values'] and data['mse_values']:
                all_data[beta] = data
            else:
                print(f"  跳过 beta={beta} 的日志，因为缺少必要的数据")
        except Exception as e:
            print(f"  处理日志文件 {log_path} 时出错: {e}")
            traceback.print_exc()
            continue
    return all_data


# --- 帕累托点提取函数 (与之前相同) ---
def extract_pareto_points(model_data):
    """
    从读取的日志数据中提取最终（收敛）的 KL 和 Recon 值。
    使用最后 10% 数据的平均值以增加稳定性。
    """
    results = {}
    for beta, data in sorted(model_data.items()):
        if not data['kl_values'] or not data['mse_values']:
            continue

        min_len = min(len(data['kl_values']), len(data['mse_values']))
        if min_len == 0:
            continue

        window_size = min(100, max(1, int(min_len * 0.1)))

        final_kl = np.mean(data['kl_values'][-window_size:])
        final_recon = np.mean(data['mse_values'][-window_size:])

        if np.isnan(final_kl) or np.isnan(final_recon):
            print(f"警告: beta={beta} 的最终值计算为 NaN，跳过")
            continue

        print(f"  Beta={beta}: 最终 KL={final_kl:.6f}, 最终 Recon={final_recon:.6f}")
        results[beta] = {'kl': final_kl, 'recon': final_recon}

    return results


# --- 新增函数：保存数据到 TXT ---
def save_data_to_txt(models_data, save_path):
    """
    将提取的帕累托数据点保存到 TXT 文件中。
    models_data: 字典，键为模型名称，值为 extract_pareto_points 返回的结果
    """
    try:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"已创建目录: {save_dir}")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# 帕累托前沿数据 (KL 散度 vs 重建损失)\n")
            f.write("# 自动生成于: " + str(np.datetime64('now')) + "\n")
            f.write("=" * 80 + "\n\n")

            for model_name, data_points in models_data.items():
                f.write(f"--- 模型: {model_name} ---\n")
                # 写入表头，使用逗号分隔 (CSV 格式)
                f.write("beta, final_kl, final_recon\n")

                # 按 beta 排序写入数据
                for beta, values in sorted(data_points.items()):
                    f.write(f"{beta}, {values['kl']:.8f}, {values['recon']:.8f}\n")

                f.write("\n" + "=" * 80 + "\n\n")

        print(f"\n数据已成功保存到: {save_path}")

    except Exception as e:
        print(f"保存文件时出错: {e}")
        traceback.print_exc()


# --- 修改后的主执行函数 ---
if __name__ == "__main__":

    # --- 1. 配置您的 runs 基础目录 ---
    # 例如: "/home/niu/workspaces/VAE_ws/agent_encoder/runs/"
    BASE_RUN_DIR = "/home/niu/workspaces/VAE_ws/agent_encoder/runs/"

    # --- 2. 配置您希望的输出文件路径 ---
    OUTPUT_FILE_PATH = "/home/niu/workspaces/VAE_ws/figure_result/pareto_plots/pareto_data.txt"

    # 确保路径以 / 结尾
    BASE_RUN_DIR = os.path.join(BASE_RUN_DIR, "")

    # --- 3. 定义两个模型的日志模式 ---
    dc_vae_pattern = os.path.join(BASE_RUN_DIR, "dc_vae_beta*_LD_64*", "**", "events.out.tfevents.*")
    beta_vae_pattern = os.path.join(BASE_RUN_DIR, "beta_vae_beta*_LD_64*", "**", "events.out.tfevents.*")

    # --- 4. 为每个模型读取日志数据 ---
    print("=" * 30 + " 读取 DC-VAE (碰撞图) 数据 " + "=" * 30)
    dc_vae_data = read_tensorboard_logs(dc_vae_pattern)

    print("\n" + "=" * 30 + " 读取 β-VAE (深度图) 数据 " + "=" * 30)
    beta_vae_data = read_tensorboard_logs(beta_vae_pattern)

    if not dc_vae_data and not beta_vae_data:
        print("\n错误: 两个模型都没有找到任何日志数据。")
        print(f"请检查 BASE_RUN_DIR 是否正确: {BASE_RUN_DIR}")
        exit(1)

    # --- 5. 为每个模型提取帕累托点（最终的 KL 和 Recon） ---
    print("\n" + "=" * 30 + " 提取 DC-VAE 帕累托点 " + "=" * 30)
    dc_vae_results = extract_pareto_points(dc_vae_data)

    print("\n" + "=" * 30 + " 提取 β-VAE 帕累托点 " + "=" * 30)
    beta_vae_results = extract_pareto_points(beta_vae_data)

    # --- 6. 准备保存的数据结构 ---
    all_models_data = {
        "DC-VAE (任务驱动: 碰撞图重构)": dc_vae_results,
        "beta-VAE (级联基线: 深度图重构)": beta_vae_results
    }

    # --- 7. 保存到 TXT 文件 ---
    print("\n" + "=" * 30 + " 正在保存数据到 TXT 文件 " + "=" * 30)
    save_data_to_txt(all_models_data, OUTPUT_FILE_PATH)

    print("\n脚本执行完毕。")