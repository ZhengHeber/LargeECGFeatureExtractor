import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import time
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.optim.lr_scheduler import StepLR
import numpy as np
import sys
from einops import rearrange

from utils import load_hdf5_file, calculate_model_size
from mamba_ssm import Mamba2
from tokenizer import CrossTokenizer
from trainPrepare import train_one_epoch_CROSS, test_one_epoch_CROSS

torch.manual_seed(49)

class ECGDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data

class ResMamba(nn.Module):
    def __init__(self, d_model=256, d_state=64, d_conv=4, dropout_rate=0.1):
        super(ResMamba, self).__init__()

        self.activation = nn.SELU()
        self.dropout = nn.AlphaDropout(dropout_rate)

        self.mamba2_block = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv)
        
    def forward(self, x):
        batch_size = x.size(0)
        # Reshape input
        x = rearrange(x, 'b (t p1) f -> b t (p1 f)', t=250, p1=20)
        x = nn.functional.pad(x, (0, 256 - 240))  # Pad to (batch_size, 250, 256)

        # Apply Mamba2 block
        x = self.mamba2_block(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Remove padding
        x = x[..., :240]  # Remove the last 16 dimensions

        # Reshape back to original dimensions
        x = rearrange(x, 'b t (p1 f) -> b (t p1) f', t=250, p1=20, f=12)

        return x

def load_file(file_path):
    try:
        data = load_hdf5_file(file_path)
        if data.shape == (5000, 12):
            if torch.isnan(data).any():
                raise ValueError(f"NaN detected in data from {file_path}")
            return data.cuda()  # Load to GPU
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

def split_dataset(all_data):
    dataset_size = len(all_data)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])
    return train_data, test_data

def load_state_dict_without_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # Initialize process group
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    loss_function = nn.MSELoss()
    model = ResMamba().to(device)
    tokenizer = CrossTokenizer().to(device)

    # Wrap model and tokenizer with DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    tokenizer = DDP(tokenizer, device_ids=[local_rank], output_device=local_rank)

    _ = calculate_model_size(model)
    _ = calculate_model_size(tokenizer)

    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.Adam(list(model.parameters()) + list(tokenizer.parameters()), lr=0.00004)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    paramDir = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Parameters')
    LogDir = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Logs')
    os.makedirs(paramDir, exist_ok=True)
    os.makedirs(LogDir, exist_ok=True)
    log_file = os.path.join(LogDir, 'TokenPretrain.log')

    with open(log_file, 'w') as log:
        log.write(f"Training started at {datetime.now()}\n")
        log.write(f"Model Structure:\n{model}\n")
        log.write(f"Tokenizer Structure:\n{tokenizer}\n")

        data_path1 = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Datasets/Mimic/Mimic')
        batch_size = 48

        file_list1 = [os.path.join(data_path1, file) for file in os.listdir(data_path1) if file.endswith('.hdf5')]
        file_list = file_list1

        num_splits = 100
        split_file_lists = [file_list[i::num_splits] for i in range(num_splits)]

        total_files = len(file_list)
        print(f"Total number of files: {total_files}")

        del file_list, file_list1

        for split_idx, split_files in enumerate(split_file_lists):
            all_data = process_all_data(split_files, device, log)
            
            train_data, test_data = split_dataset(all_data)

            train_dataset = ECGDataset(train_data)
            test_dataset = ECGDataset(test_data)

            train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            del all_data, train_data, test_data
            torch.cuda.empty_cache()

            num_epochs = 20
            best_test_loss = float('inf')

            training_start_time = time.time()

            for epoch in range(num_epochs):
                epoch_start_time = time.time()

                with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}, Split {split_idx+1} Training", disable=(dist.get_rank() != 0)) as pbar_train:
                    train_epoch_loss = train_one_epoch_CROSS(model, tokenizer, train_data_loader, loss_function, optimizer, device, pbar=pbar_train)

                with tqdm(total=len(test_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}, Split {split_idx+1} Testing", disable=(dist.get_rank() != 0)) as pbar_test:
                    test_epoch_loss = test_one_epoch_CROSS(model, tokenizer, test_data_loader, loss_function, device, pbar=pbar_test)

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time

                print(f"Epoch {epoch+1}, Split {split_idx+1}, Train Loss: {train_epoch_loss}, Test Loss: {test_epoch_loss}, Duration: {epoch_duration:.2f}s")
                log.write(f"Epoch {epoch+1}, Split {split_idx+1}, Train Loss: {train_epoch_loss}, Test Loss: {test_epoch_loss}, Duration: {epoch_duration:.2f}s\n")

                test_loss_tensor = torch.tensor(test_epoch_loss, device=device)
                dist.reduce(test_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
                avg_test_loss = test_loss_tensor.item() / dist.get_world_size()

                log.write(f"Average Test Loss for Epoch {epoch+1}, Split {split_idx+1}: {avg_test_loss}\n")

                if dist.get_rank() == 0 and avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_model_path = os.path.join(paramDir, f'best_CrossTokenizer_model_split_{split_idx+1}.pth')
                    torch.save({'model_state_dict': model.state_dict(), 'tokenizer_state_dict': tokenizer.state_dict()}, best_model_path)
                    print(f"Saved new best model for split {split_idx+1} with Test Loss: {avg_test_loss}")
                    log.write(f"Saved new best model for split {split_idx+1} with Test Loss: {avg_test_loss}\n")

                scheduler.step()

            del train_data_loader, test_data_loader
            torch.cuda.empty_cache()

            if dist.get_rank() == 0:
                final_model_path = os.path.join(paramDir, f'final_CrossTokenizer_model_split_{split_idx+1}.pth')
                torch.save({'model_state_dict': model.state_dict(), 'tokenizer_state_dict': tokenizer.state_dict()}, final_model_path)
                print(f"Saved final model for split {split_idx+1}.")
                log.write(f"Saved final model for split {split_idx+1}.\n")

            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time
            print(f"Total training time for split {split_idx+1}: {total_training_time:.2f}s")
            log.write(f"Total training time for split {split_idx+1}: {total_training_time:.2f}s\n")

            if split_idx < 99 and dist.get_rank() == 0:
                best_model_path = os.path.join(paramDir, f'best_CrossTokenizer_model_split_{split_idx+1}.pth')
                final_model_path = os.path.join(paramDir, f'final_CrossTokenizer_model_split_{split_idx+1}.pth')
                
                try:
                    checkpoint = torch.load(best_model_path)
                    print(f"Loaded best model parameters from split {split_idx+1} for split {split_idx+2}.")
                    log.write(f"Loaded best model parameters from split {split_idx+1} for split {split_idx+2}.\n")
                except FileNotFoundError:
                    checkpoint = torch.load(final_model_path)
                    print(f"Best model not found. Loaded final model parameters from split {split_idx+1} for split {split_idx+2}.")
                    log.write(f"Best model not found. Loaded final model parameters from split {split_idx+1} for split {split_idx+2}.\n")
                
                model.load_state_dict(checkpoint['model_state_dict'])
                tokenizer.load_state_dict(checkpoint['tokenizer_state_dict'])

        log.write(f"Training completed at {datetime.now()}\n")

if __name__ == "__main__":
    main()
