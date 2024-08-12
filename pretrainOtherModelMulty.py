import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from tokenizer import NullTokenizer  # 使用新的tokenizer
from utils import random_masking, normalize, load_hdf5_file
from trainPrepare import train_one_epoch_simple as train_one_epoch, test_one_epoch_simple as test_one_epoch
from mamba_ssm import Mamba2

torch.manual_seed(45)

class SelfSupervisedMamba2(nn.Module):
    def __init__(self, d_model=256, d_state=64, d_conv=4, expand=2):
        super(SelfSupervisedMamba2, self).__init__()
        self.mamba2 = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.LeakyReLU = nn.LeakyReLU()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.mamba2(x)
        x = self.LeakyReLU(x)
        x = self.linear(x)
        x = self.dropout(x)
        # x = self.mamba2(x)
        # x = self.LeakyReLU(x)
        # x = self.linear(x)
        # x = self.dropout(x)
        x = self.mamba2(x)
        x = self.linear(x)
        return x

def load_file(file_path):
    try:
        data = load_hdf5_file(file_path)
        if data.shape == (5000, 12):
            return data.cuda()  # 加载到 GPU
    except Exception as e:
        return f"Error loading {file_path}: {e}"
    return None

def process_all_data(split_files, device, log):
    all_data = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        future_to_file = {executor.submit(load_file, file_path): file_path for file_path in split_files}
        for future in tqdm(as_completed(future_to_file), total=len(split_files), desc="Loading data", disable=(dist.get_rank() != 0)):
            file_path = future_to_file[future]
            try:
                data = future.result()
                if isinstance(data, str):  # Check if it's an error message
                    print(data)
                    log.write(data + "\n")
                elif data is not None:
                    all_data.append(data)
            except Exception as e:
                error_message = f"Error processing {file_path}: {e}"
                print(error_message)
                log.write(error_message + "\n")

    if all_data:
        print(f'数据类型: {all_data[0].dtype}')
        log.write(f'数据类型: {all_data[0].dtype}\n')

    return all_data

def process_dataset(all_data, batch_size, device, log):
    num_batches = len(all_data) // batch_size
    remainder = len(all_data) % batch_size

    dataset = [torch.stack(all_data[i * batch_size:(i + 1) * batch_size]).float() for i in range(num_batches)]

    if remainder > 0:
        print('Last batch dropped')
        log.write('Last batch dropped\n')

    all_data = torch.cat(dataset, dim=0).float()
    mean = all_data.mean(dim=(0, 1))
    std = all_data.std(dim=(0, 1))

    dataset = [(normalize(data, mean, std)) for data in dataset]

    if dataset:
        print(f'数据形状: {dataset[0].shape}')
        log.write(f'数据形状: {dataset[0].shape}\n')

    return dataset

def split_dataset(dataset, log):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def tokenize_data_list(data_list, tokenizer, device, desc, log):
    tokenized_data_list = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        future_to_data = {executor.submit(tokenizer.tokenize, data.to(device)): data for data in data_list}
        for future in tqdm(as_completed(future_to_data), desc=desc, total=len(data_list), disable=(dist.get_rank() != 0)):
            data = future_to_data[future]
            try:
                tokenized_data = future.result()
                tokenized_data_list.append(tokenized_data)
            except Exception as e:
                error_message = f"Error tokenizing data {data}: {e}"
                print(error_message)
                log.write(error_message + "\n")
    return tokenized_data_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # 初始化进程组
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    loss_function = nn.MSELoss()
    tokenizer = NullTokenizer()  # 不需要放到device上
    model = SelfSupervisedMamba2().to(device)

    # 使用DistributedDataParallel包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 只优化模型的参数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    paramDir = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Parameters')
    LogDir = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Logs')
    os.makedirs(paramDir, exist_ok=True)
    os.makedirs(LogDir, exist_ok=True)
    log_file = os.path.join(LogDir, 'ECGModelPretrain.log')

    with open(log_file, 'w') as log:
        log.write(f"Training started at {datetime.now()}\n")
        log.write(f"Model Structure:\n{model}\n")
        log.write(f"Tokenizer Structure:\n{tokenizer}\n")

        data_path1 = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Datasets/Mimic/Mimic')
        data_path2 = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Datasets/wfdblead')
        batch_size = 32

        # 获取两个路径中的所有文件
        file_list1 = [os.path.join(data_path1, file) for file in os.listdir(data_path1) if file.endswith('.hdf5')]
        file_list2 = [os.path.join(data_path2, file) for file in os.listdir(data_path2) if file.endswith('.hdf5')]

        # 合并文件列表
        file_list = file_list1 + file_list2

        num_splits = 50
        split_file_lists = [file_list[i::num_splits] for i in range(num_splits)]

        # 输出总文件数量
        total_files = len(file_list)
        print(f"Total number of files: {total_files}")

        del file_list, file_list1, file_list2

        for split_idx, split_files in enumerate(split_file_lists):
            all_data = process_all_data(split_files, device, log)
            dataset = process_dataset(all_data, batch_size, device, log)
            del all_data
            torch.cuda.empty_cache()

            train_dataset, test_dataset = split_dataset(dataset, log)
            del dataset
            torch.cuda.empty_cache()

            train_tokenized_data_list = tokenize_data_list(train_dataset, tokenizer, device, "Tokenizing train dataset", log)
            test_tokenized_data_list = tokenize_data_list(test_dataset, tokenizer, device, "Tokenizing test dataset", log)

            print(f"First element shape in train_tokenized_data_list: {train_tokenized_data_list[0].shape}")
            print(f"First element shape in test_tokenized_data_list: {test_tokenized_data_list[0].shape}")

            num_epochs = 10
            mask_ratio = 0.5
            best_test_loss = float('inf')

            training_start_time = time.time()

            train_data_loader = DataLoader(train_tokenized_data_list, batch_size=1, shuffle=True)
            test_data_loader = DataLoader(test_tokenized_data_list, batch_size=1, shuffle=False)
            del train_tokenized_data_list, test_tokenized_data_list
            torch.cuda.empty_cache()

            for epoch in range(num_epochs):
                epoch_start_time = time.time()

                with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}, Split {split_idx+1} Training", disable=(dist.get_rank() != 0)) as pbar_train:
                    train_epoch_loss = train_one_epoch(model, tokenizer, train_data_loader, loss_function, optimizer, mask_ratio, device, device, pbar=pbar_train)

                with tqdm(total=len(test_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}, Split {split_idx+1} Testing", disable=(dist.get_rank() != 0)) as pbar_test:
                    test_epoch_loss = test_one_epoch(model, tokenizer, test_data_loader, loss_function, mask_ratio, device, device, pbar=pbar_test)

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time

                print(f"Epoch {epoch+1}, Split {split_idx+1}, Train Loss: {train_epoch_loss}, Test Loss: {test_epoch_loss}, Duration: {epoch_duration:.2f}s")
                log.write(f"Epoch {epoch+1}, Split {split_idx+1}, Train Loss: {train_epoch_loss}, Test Loss: {test_epoch_loss}, Duration: {epoch_duration:.2f}s\n")

                # 通过分布式操作减少所有进程的测试损失
                test_loss_tensor = torch.tensor(test_epoch_loss, device=device)
                dist.reduce(test_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
                avg_test_loss = test_loss_tensor.item() / dist.get_world_size()

                log.write(f"Average Test Loss for Epoch {epoch+1}, Split {split_idx+1}: {avg_test_loss}\n")

                if dist.get_rank() == 0 and avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_model_path = os.path.join(paramDir, f'best_PureMamba2_model_split_{split_idx+1}.pth')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                    }, best_model_path)
                    print(f"Saved new best model for split {split_idx+1} with Test Loss: {avg_test_loss}")
                    log.write(f"Saved new best model for split {split_idx+1} with Test Loss: {avg_test_loss}\n")

            if dist.get_rank() == 0:
                final_model_path = os.path.join(paramDir, f'final_PureMamba2_model_split_{split_idx+1}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, final_model_path)
                print(f"Saved final model for split {split_idx+1}.")
                log.write(f"Saved final model for split {split_idx+1}.\n")

            del train_data_loader, test_data_loader
            del train_dataset, test_dataset
            torch.cuda.empty_cache()

            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time
            print(f"Total training time for split {split_idx+1}: {total_training_time:.2f}s")
            log.write(f"Total training time for split {split_idx+1}: {total_training_time:.2f}s\n")

            if split_idx < 49 and dist.get_rank() == 0:
                best_model_path = os.path.join(paramDir, f'best_PureMamba2_model_split_{split_idx+1}.pth')
                final_model_path = os.path.join(paramDir, f'final_PureMamba2_model_split_{split_idx+1}.pth')
                
                try:
                    checkpoint = torch.load(best_model_path)
                    print(f"Loaded best model parameters from split {split_idx+1} for split {split_idx+2}.")
                    log.write(f"Loaded best model parameters from split {split_idx+1} for split {split_idx+2}.\n")
                except FileNotFoundError:
                    checkpoint = torch.load(final_model_path)
                    print(f"Best model not found. Loaded final model parameters from split {split_idx+1} for split {split_idx+2}.")
                    log.write(f"Best model not found. Loaded final model parameters from split {split_idx+1} for split {split_idx+2}.\n")
                
                model.load_state_dict(checkpoint['model_state_dict'])

        log.write(f"Training completed at {datetime.now()}\n")

if __name__ == "__main__":
    main()
