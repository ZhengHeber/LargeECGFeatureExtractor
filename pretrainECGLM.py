import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange
import os
import h5py
import time
from datetime import datetime

from tokenizer import ECGMetaTokenizer, ECGSpaceTokenizer
from utils import random_masking, tokenize_data_space, normalize, load_hdf5_file
from trainPrepare import (
    train_one_epoch_ECGLM as train_one_epoch,
    test_one_epoch_ECGLM as test_one_epoch
)
from mamba_ssm import Mamba, Mamba2

torch.manual_seed(41)

class SelfSupervisedMamba2(nn.Module):
    """
    This class defines a self-supervised Mamba2 model with a linear layer.
    """
    def __init__(self, d_model=256, d_state=64, d_conv=4, expand=2):
        super(SelfSupervisedMamba2, self).__init__()
        self.mamba2 = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.mamba2(x)
        x = self.linear(x)
        return x

class SelfSupervisedMamba(nn.Module):
    """
    This class defines a self-supervised Mamba model with a linear layer.
    """
    def __init__(self, d_model=256, d_state=64, d_conv=4, expand=2):
        super(SelfSupervisedMamba, self).__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.mamba(x)
        x = self.linear(x)
        return x

class CompleteModel(nn.Module):
    """
    This class combines the self-supervised Mamba2 model and the space tokenizer.
    """
    def __init__(self, model, space_tokenizer):
        super(CompleteModel, self).__init__()
        self.model = model
        self.space_tokenizer = space_tokenizer

    def forward(self, x):
        x = self.space_tokenizer(x)
        x = self.model(x)
        return x

# 设置要使用的设备
model_device = torch.device('cuda:0')
data_device = torch.device('cuda:0')

# 3. 定义损失函数和优化器
loss_function = nn.MSELoss()
meta_tokenizer = ECGMetaTokenizer()
space_tokenizer = ECGSpaceTokenizer().to(model_device)
model = SelfSupervisedMamba2().to(model_device)

optimizer = optim.Adam(list(model.parameters()) + list(space_tokenizer.parameters()), lr=0.001)

# 设置参数保存目录和日志文件路径
paramDir = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Parameters')
LogDir = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Logs')
os.makedirs(paramDir, exist_ok=True)
os.makedirs(LogDir, exist_ok=True)
log_file = os.path.join(LogDir, 'ImpliedMamba2Pretrain.log')

# 创建完整模型并打印结构
complete_model = CompleteModel(model, space_tokenizer).to(model_device)
print(complete_model)

# 释放完整模型
del complete_model

# 打开日志文件
with open(log_file, 'w') as log:
    log.write(f"Training started at {datetime.now()}\n")
    log.write(f"Model Structure:\n{model}\n")
    log.write(f"Space Tokenizer Structure:\n{space_tokenizer}\n")

    # 数据路径
    data_path = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Datasets/SampleHDF5')
    batch_size = 64  # 你可以根据需要调整批次大小

    # 获取所有HDF5文件路径
    file_list = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.hdf5')]

    # 获取文件总数并分成十份
    num_files = len(file_list)
    files_per_batch = num_files // 10
    remainder_files = num_files % 10

    all_data = []
    for i in range(10):
        start_idx = i * files_per_batch
        end_idx = start_idx + files_per_batch
        batch_files = file_list[start_idx:end_idx]

        batch_data = []
        for file_path in batch_files:
            data = load_hdf5_file(file_path)
            if data.shape == (5000, 12):
                batch_data.append(data)

        if batch_data:
            all_data.extend(batch_data)

        # 查看读入的真实数据的数据类型
        if batch_data:
            print(f'数据类型: {batch_data[0].dtype}')
            log.write(f'数据类型: {batch_data[0].dtype}\n')

        # 将数据整合成批次
        num_batches = len(batch_data) // batch_size
        dataset = [torch.stack(batch_data[i * batch_size:(i + 1) * batch_size]).float() for i in range(num_batches)]

        all_data = torch.cat(dataset, dim=0).float()  # 将所有batch连接起来并转换为浮点型
        mean = all_data.mean(dim=(0, 1))  # 计算全局均值
        std = all_data.std(dim=(0, 1))    # 计算全局标准差

        dataset = [(normalize(data, mean, std)) for data in dataset]

        # 由于 dataset 是列表，无法直接调用 shape，需要对列表中的第一个张量调用 shape
        if dataset:
            print(f'数据形状: {dataset[0].shape}')
            log.write(f'数据形状: {dataset[0].shape}\n')

        # 分割训练集和测试集
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_tokenized_data_list_time = [meta_tokenizer.tokenize(data.to(data_device)) for data in train_dataset]
        test_tokenized_data_list_time = [meta_tokenizer.tokenize(data.to(data_device)) for data in test_dataset]

        num_epochs = 300
        mask_ratio = 0.5
        best_test_loss = float('inf')

        # 开始训练
        training_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            train_tokenized_data_list_all = tokenize_data_space(space_tokenizer, train_dataset, train_tokenized_data_list_time, data_device)
            test_tokenized_data_list_all = tokenize_data_space(space_tokenizer, test_dataset, test_tokenized_data_list_time, data_device)

            train_data_loader = DataLoader(train_tokenized_data_list_all, batch_size=1, shuffle=True)
            test_data_loader = DataLoader(test_tokenized_data_list_all, batch_size=1, shuffle=False)

            train_epoch_loss = train_one_epoch(model, meta_tokenizer, space_tokenizer, train_data_loader, loss_function, optimizer, mask_ratio, model_device, data_device)
            test_epoch_loss = test_one_epoch(model, meta_tokenizer, space_tokenizer, test_data_loader, loss_function, mask_ratio, model_device, data_device)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"Epoch {epoch+1}, Train Loss: {train_epoch_loss}, Test Loss: {test_epoch_loss}, Duration: {epoch_duration:.2f}s")
            log.write(f"Epoch {epoch+1}, Train Loss: {train_epoch_loss}, Test Loss: {test_epoch_loss}, Duration: {epoch_duration:.2f}s\n")

            # 保存测试集损失值最低的模型参数
            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                best_model_path = os.path.join(paramDir, 'best_ImpliedMamba2_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model with Test Loss: {test_epoch_loss}")
                log.write(f"Saved new best model with Test Loss: {test_epoch_loss}\n")

    # 训练完成后保存最终模型参数
    final_model_path = os.path.join(paramDir, 'final_ImpliedMamba2_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print("Saved final model.")
    log.write("Saved final model.\n")

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"Total training time: {total_training_time:.2f}s")
    log.write(f"Total training time: {total_training_time:.2f}s\n")
    log.write(f"Training completed at {datetime.now()}\n")
