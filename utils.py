import torch
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import torch.distributed as dist

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, B, D, L], sequence
    """
    N, B, D, L = x.shape
    len_keep = int(D * (1 - mask_ratio))
    noise = torch.rand(N, B, D, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=2)
    ids_restore = torch.argsort(ids_shuffle, dim=2)
    ids_keep = ids_shuffle[:, :, :len_keep]
    mask = torch.ones([N, B, D], device=x.device)
    mask[:, :, :len_keep] = 0
    mask = torch.gather(mask, dim=2, index=ids_restore)
    x_masked = x.clone()
    x_masked[mask.unsqueeze(-1).repeat(1, 1, 1, L) == 1] = 0
    return x_masked, mask.to(torch.bool)

def tokenize_data_space(tokenizer, data_list, data_list_padded, data_device):
    """
    Tokenizes the data by passing data_list_padded to the space_tokenizer.
    """
    tokenized_data_list = []
    for i in range(len(data_list)):
        data_padded = data_list_padded[i].to(data_device)
        tokenized_data = tokenizer.tokenize(data_list[i].to(data_device), data_padded)
        tokenized_data_list.append(tokenized_data)
    return tokenized_data_list

def normalize(data, mean, std):
    mean = mean.to(data.device)  # 确保均值在同一设备上
    std = std.to(data.device)    # 确保标准差在同一设备上
    return (data - mean) / std


def denormalize(data, mean, std):
    mean = mean.to(data.device)  # 确保均值在同一设备上
    std = std.to(data.device)    # 确保标准差在同一设备上
    return data * std + mean

def load_hdf5_file(file_path):
    """
    Loads HDF5 file and returns the dataset.
    """
    with h5py.File(file_path, 'r') as f:
        data = f['signal'][:]  # Assuming dataset is stored in 'data' dataset
    return torch.tensor(data, dtype=torch.float32)

def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    param_size_mb = total_params * 4 / (1024 ** 2)  # 参数大小 (MB), assuming each parameter is a float32 (4 bytes)
    if param_size_mb < 1024:
        print(f"Total model size: {param_size_mb:.2f} MB")
    else:
        param_size_gb = param_size_mb / 1024
        print(f"Total model size: {param_size_gb:.2f} GB")
    return total_params, param_size_mb

def load_state_dict_without_module_prefix(state_dict):
    """
    Remove 'module.' prefix from keys in state_dict if present.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

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


