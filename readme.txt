
训练文件：train_base.py
测试pth模型：eval_pth.py
将pth模型转换为onnx：to_onnx.py
测试onnx模型：eval_onnx.py

在训练时，
在train_base.py中可以设置1）训练数据路径 datapath，为不同文件夹路径的list；2）模型保存路径 checkpoint_path；3）模型结构 --net；4）损失函数 --loss_type。
在global_settings.py中选择 mean和std。



在测试时，
在eval_pth.py中可以设置1）测试数据路径 source_path；2）模型路径model_path；3）模型结构 -net；4）在get_test_dataloader中设置class_number和num_per，分别对应测试图像类别数和每个类别图像数