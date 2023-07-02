- 配置：在run_nerf.py的config_parser函数中可以设置实验的参数。
- 训练：在命令行运行python run_nerf.py --config configs/bicycle.txt，训练好的最优模型将会保存至./trained_model。
- 测试：把200000.tar文件放入logs/bicycle_test文件夹中，在命令行运行python run_nerf.py --config configs/bicycle.txt，即可得到视频


由于电脑内存有限，因此在构建数据集时使用了较低分辨率的图片，得到的视频较为模糊。若使用较高分辨率的图片，如fern数据集中的图片，则经过训练可以得到较高分辨率的重建结果。