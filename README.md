# VAE4AerialGym

This repository is an implement of the paper "Task-Driven Compression for Collision  Encoding Based on Depth Images"

该库是“基于深度图像的碰撞编码任务驱动压缩”论文的实现



你需要在`VAE4AerialGym\datasets`中准备你的数据集，原始深度图存储于`VAE4AerialGym\datasets\depths`

准备好深度图后 生成碰撞图使用脚本（这里还需要优化，可能会花费您半天的时间，取决于你的深度图数据集大小和电脑性能）：

```bash
python agent_encoder/Project_dataset_generate.py
```

生成的碰撞图位于`VAE4AerialGym\datasets\colls_offset`

训练代码使用脚本：

```bash
python agent_encoder/train_betaVAE.py
```

测试时，需要指定模型，然后使用脚本：

```bash
python agent_encoder/vae_image_a_test.py
python agent_encoder/vae_image_batch_test.py
```

