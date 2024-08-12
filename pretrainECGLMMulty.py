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
from torch.optim.lr_scheduler import StepLR


from tokenizer import CrossTokenizer, DeviatTokenizer
from utils import random_masking, normalize, load_hdf5_file, calculate_model_size, load_state_dict_without_module_prefix, process_all_data, split_dataset, process_dataset, tokenize_data_list
from trainPrepare import train_one_epoch_ECGLM_model, test_one_epoch_ECGLM_model
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
        x = self.dropout(x)
        x = self.linear(x)
        x = self.mamba2(x)
        x = self.linear(x)
        return x

class ResMamba(nn.Module):
    def __init__(self, d_model=256, d_state=64, d_conv=4, expand=2, depth=4, dropout_rate=0.1):
        super(ResMamba, self).__init__()
        self.depth = depth

        self.mamba2_blocks_up = nn.ModuleList()
        self.mamba2_blocks_down = nn.ModuleList()
        self.proj_layers_up = nn.ModuleList()
        self.proj_layers_down = nn.ModuleList()
        self.activation = nn.SELU()
        self.dropout = nn.AlphaDropout(dropout_rate)

        current_d_model = d_model
        for _ in range(depth):
            self.mamba2_blocks_up.append(Mamba2(d_model=current_d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            next_d_model = current_d_model * expand
            self.proj_layers_up.append(nn.Sequential(
                nn.Conv1d(current_d_model, next_d_model, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(next_d_model),
                self.activation,
                self.dropout
            ))
            current_d_model = next_d_model

        for _ in range(depth):
            next_d_model = current_d_model // expand
            self.proj_layers_down.append(nn.Sequential(
                nn.Conv1d(current_d_model, next_d_model, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(next_d_model),
                self.activation,
                self.dropout
            ))
            self.mamba2_blocks_down.append(Mamba2(d_model=next_d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            current_d_model = next_d_model

    def forward(self, x):
        residuals = []
        for mamba_block, proj_layer in zip(self.mamba2_blocks_up, self.proj_layers_up):
            x = mamba_block(x)
            x = x.transpose(1, 2)
            x = proj_layer(x)
            x = x.transpose(1, 2)
            residuals.append(x)

        for proj_layer, mamba_block in zip(self.proj_layers_down, self.mamba2_blocks_down):
            if residuals:
                res = residuals.pop()
                x = x + res
            x = x.transpose(1, 2)
            x = proj_layer(x)
            x = x.transpose(1, 2)
            x = mamba_block(x)

        return x

def freeze_parameters(parameters, freeze=True):
    for param in parameters:
        param.requires_grad = not freeze

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    loss_function = nn.MSELoss()
    assoc_tokenizer = CrossTokenizer().to(device)
    devia_tokenizer = DeviatTokenizer().to(device)
    model = ResMamba(
        d_model=256,
        d_state=128,
        d_conv=4,
        expand=2,
        depth=2,
    ).to(device)

    _ = calculate_model_size(model)

    Associatetokenizer_checkpoint = torch.load('/public/home/wangzeyu/zhenghb/ECGLLM/Parameters/best_CrossTokenizer_model_split_100.pth', map_location=device)
    Deviationtokenizer_checkpoint = torch.load('/public/home/wangzeyu/zhenghb/ECGLLM/Parameters/best_DeviatTokenizer_model_split_100.pth', map_location=device)

    Assoc_tokenizer_state_dict = load_state_dict_without_module_prefix(Associatetokenizer_checkpoint['tokenizer_state_dict'])
    assoc_tokenizer.load_state_dict(Assoc_tokenizer_state_dict)

    Devia_tokenizer_state_dict = load_state_dict_without_module_prefix(Deviationtokenizer_checkpoint['tokenizer_state_dict'])
    devia_tokenizer.load_state_dict(Devia_tokenizer_state_dict)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    assoc_tokenizer = DDP(assoc_tokenizer, device_ids=[local_rank], output_device=local_rank)
    devia_tokenizer = DDP(devia_tokenizer, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    paramDir = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Parameters')
    LogDir = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Logs')
    os.makedirs(paramDir, exist_ok=True)
    os.makedirs(LogDir, exist_ok=True)
    log_file = os.path.join(LogDir, 'ECGModelPretrain.log')

    with open(log_file, 'w') as log:
        log.write(f"Training started at {datetime.now()}\n")
        log.write(f"Model Structure:\n{model}\n")
        log.write(f"Meta Tokenizer Structure:\n{assoc_tokenizer}\n")
        log.write(f"Space Tokenizer Structure:\n{devia_tokenizer}\n")

        data_path1 = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Datasets/Mimic/Mimic')
        data_path2 = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Datasets/wfdblead')
        data_path3 = os.path.expanduser('/public/home/wangzeyu/zhenghb/ECGLLM/Datasets/TJdata')
        batch_size = 128

        file_list1 = [os.path.join(data_path1, file) for file in os.listdir(data_path1) if file.endswith('.hdf5')]
        file_list2 = [os.path.join(data_path2, file) for file in os.listdir(data_path2) if file.endswith('.hdf5')]
        file_list3 = [os.path.join(data_path3, file) for file in os.listdir(data_path2) if file.endswith('.hdf5')]

        file_list = file_list1 + file_list2 + file_list3

        num_splits = 50
        split_file_lists = [file_list[i::num_splits] for i in range(num_splits)]

        total_files = len(file_list)
        print(f"Total number of files: {total_files}")
        del file_list, file_list1

        # 初始布尔值
        b = True

        for split_idx, split_files in enumerate(split_file_lists):
            all_data = process_all_data(split_files, device, log)
            dataset = process_dataset(all_data, batch_size, device, log)
            del all_data
            torch.cuda.empty_cache()

            train_dataset, test_dataset = split_dataset(dataset, log)
            del dataset
            torch.cuda.empty_cache()

            train_tokenized_data_list_cross = tokenize_data_list(train_dataset, assoc_tokenizer, device, "Tokenizing train dataset with crosstokenizer", log)
            test_tokenized_data_list_cross = tokenize_data_list(test_dataset, assoc_tokenizer, device, "Tokenizing test dataset with crosstokenizer", log)

            train_tokenized_data_list_devia = tokenize_data_list(train_dataset, devia_tokenizer, device, "Tokenizing train dataset with deviatokenizer", log)
            test_tokenized_data_list_devia = tokenize_data_list(test_dataset, devia_tokenizer, device, "Tokenizing train dataset with deviatokenizer", log)

            train_tokenized_data_list_all = torch.cat([train_dataset, train_tokenized_data_list_cross, train_tokenized_data_list_devia], dim=1)
            test_tokenized_data_list_all = torch.cat([train_dataset, train_tokenized_data_list_devia, test_tokenized_data_list_devia], dim=1)

            del train_dataset, test_dataset
            del train_tokenized_data_list_cross, test_tokenized_data_list_cross, train_tokenized_data_list_devia, test_tokenized_data_list_devia
            torch.cuda.empty_cache()

            train_data_loader = DataLoader(train_tokenized_data_list_all, batch_size=1, shuffle=True)
            test_data_loader = DataLoader(test_tokenized_data_list_all, batch_size=1, shuffle=False)

            del train_tokenized_data_list_all, test_tokenized_data_list_all
            torch.cuda.empty_cache()

            num_epochs = 50
            mask_ratio = 0.5
            best_test_loss = float('inf')

            training_start_time = time.time()

            for epoch in range(num_epochs):
                epoch_start_time = time.time()

                # 根据布尔值冻结参数
                if b:
                    freeze_parameters(model.parameters(), freeze=True)
                    freeze_parameters(assoc_tokenizer.parameters(), freeze=False)
                    freeze_parameters(devia_tokenizer.parameters(), freeze=False)
                else:
                    freeze_parameters(model.parameters(), freeze=False)
                    freeze_parameters(assoc_tokenizer.parameters(), freeze=True)
                    freeze_parameters(devia_tokenizer.parameters(), freeze=True)

                with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}, Split {split_idx+1} Training", disable=(dist.get_rank() != 0)) as pbar_train:
                    train_epoch_loss = train_one_epoch_ECGLM_model(model, assoc_tokenizer, devia_tokenizer, train_data_loader, loss_function, optimizer, mask_ratio, device, device, pbar=pbar_train)

                with tqdm(total=len(test_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}, Split {split_idx+1} Testing", disable=(dist.get_rank() != 0)) as pbar_test:
                    test_epoch_loss = test_one_epoch_ECGLM_model(model, assoc_tokenizer, devia_tokenizer, test_data_loader, loss_function, mask_ratio, device, device, pbar=pbar_test)

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
                    best_model_path = os.path.join(paramDir, f'best_TokenizedMamba2_model_split_{split_idx+1}.pth')
                    torch.save({
                        'model_state_dict': model.state_dict()
                    }, best_model_path)
                    print(f"Saved new best model for split {split_idx+1} with Test Loss: {avg_test_loss}")
                    log.write(f"Saved new best model for split {split_idx+1} with Test Loss: {avg_test_loss}\n")

                scheduler.step()

                # 翻转布尔值
                b = not b

            del train_data_loader, test_data_loader
            torch.cuda.empty_cache()
            
            if dist.get_rank() == 0:
                final_model_path = os.path.join(paramDir, f'final_TokenizedMamba2_model_split_{split_idx+1}.pth')
                torch.save({
                    'assoc_state_dict': assoc_tokenizer.state_dict(),
                    'devia_state_dict': devia_tokenizer.state_dict()
                }, final_model_path)
                print(f"Saved final model for split {split_idx+1}.")
                log.write(f"Saved final model for split {split_idx+1}.\n")

            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time
            print(f"Total training time for split {split_idx+1}: {total_training_time:.2f}s")
            log.write(f"Total training time for split {split_idx+1}: {total_training_time:.2f}s\n")

            if split_idx < 79 and dist.get_rank() == 0:
                best_model_path = os.path.join(paramDir, f'best_TokenizedMamba2_model_split_{split_idx+1}.pth')
                final_model_path = os.path.join(paramDir, f'final_TokenizedMamba2_model_split_{split_idx+1}.pth')
                
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
