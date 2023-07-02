- 配置：在./configs中可以设置实验的参数、数据变换的方式。
- 自监督预训练：运行./trainstage1.py，最终得到预训练好的模型model_stage1_best.pth。
- 微调：运行./trainstage2.py，最终得到微调好的模型model_stage2_best.pth。
- 测试：运行./eval.py，调用训练好的模型在测试集上检验模型效果。

