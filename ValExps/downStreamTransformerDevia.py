import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import h5py
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from einops import rearrange
from mamba_ssm import Mamba2, Mamba
from tokenizer import DeviatTokenizer


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc1_input_dim = self._get_conv_output_dim()
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def _get_conv_output_dim(self):
        dummy_input = torch.zeros(1, 12, 5000).to(device)
        dummy_input = dummy_input.to(next(self.parameters()).device)
        x = self.layer1(dummy_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes=4, d_model=512, nhead=4, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.initial_conv = nn.Conv1d(12, d_model, kernel_size=1, stride=1, bias=False).to(device)
        self.final_conv = nn.Conv1d(d_model, 12, kernel_size=1, stride=1, bias=False).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout).to(device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers).to(device)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, d_model)).to(device)
        self.fc1_input_dim = self._get_trans_output_dim()
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def _get_trans_output_dim(self):
        dummy_input = torch.zeros(1, 5000, 24)
        x = dummy_input.to(next(self.parameters()).device)
        x = rearrange(x, 'b (t p1) f -> b t (p1 f)', t=250, p1=20)
        x = nn.functional.pad(x, (0, 512 - 480))  # Pad to (batch_size, 250, 256)

        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)

        x = x[..., :480]  # Remove the last 16 dimensions
        x = rearrange(x, 'b t (p1 f) -> b (t p1) f', t=250, p1=20, f=24)

        x = rearrange(x, 'b t f -> b f t')
        return int(np.prod(x.size()))

    def forward(self, x):
        x = rearrange(x, 'b (t p1) f -> b t (p1 f)', t=250, p1=20)
        x = nn.functional.pad(x, (0, 512 - 480))  # Pad to (batch_size, 250, 256)

        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)

        x = x[..., :480]  # Remove the last 16 dimensions
        x = rearrange(x, 'b t (p1 f) -> b (t p1) f', t=250, p1=20, f=24)

        x = rearrange(x, 'b t f -> b f t')

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class MambaClass(nn.Module):
    def __init__(self, d_model=512, d_state=64, d_conv=4, num_classes=4, dropout_rate=0.1):
        super(MambaClass, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.activation = nn.SELU()
        self.dropout = nn.AlphaDropout(dropout_rate)

        # self.mamba2_block = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv).to(self.device)
        self.mamba_block = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv).to(self.device)

        self.fc1_input_dim = self._get_mamba2_output_dim()
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
    def _get_mamba2_output_dim(self):
        dummy_input = torch.zeros(1, 5000, 24).to(self.device)
        dummy_input = dummy_input.to(next(self.parameters()).device)
        x = self._apply_mamba(dummy_input)
        return int(np.prod(x.size()))

    def _apply_mamba(self, x):
        # Ensure x is on the correct device
        x = x.to(self.device)

        # Reshape input
        x = rearrange(x, 'b (t p1) f -> b t (p1 f)', t=250, p1=20)
        x = nn.functional.pad(x, (0, 512 - 480))  # Pad to (batch_size, 250, 256)

        # Apply Mamba2 block
        x = self.mamba_block(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Remove padding
        x = x[..., :480]  # Remove the last 16 dimensions

        # Reshape back to original dimensions
        x = rearrange(x, 'b t (p1 f) -> b (t p1) f', t=250, p1=20, f=24)

        x = rearrange(x, 'b t f -> b f t')

        return x

    def forward(self, x):
        x = x.to(self.device)
        x = self._apply_mamba(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

num_epochs = 10
batch_size = 32
learning_rate = 0.0001
num_classes = 8  # CD, HYP, STTC, MI

# Paths
normecg_file_path = os.path.expanduser('~/HOME/ECGLM/Validations/average_norm_ecg_signal_10.npy')
label_file_path = os.path.expanduser('~/HOME/ECGLM/Validations/ecg_metadata_label.csv')
data_path = os.path.expanduser('~/HOME/ECGLM/Validations/records')

# Load labels
print("Loading labels...")
df = pd.read_csv(label_file_path)
df = df[['ECG_ID', 'Rhythms', 'Electric Axis of the Heart', 'Conduction Abnormalities',
         'Extrasystolies', 'Hypertrophies', 'Ischemia', 'Non-Specific Repolarization Abnormalities', 'Other States']]

average_norm_ecg_signal_10 = torch.tensor(np.load(normecg_file_path), dtype=torch.float32).to(device)

def load_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['ecg'][:]
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(1, 0)
    return data

def load_and_process_data(idx, df, data_path, normecg, deviat_tokenizer):
    file_name = df.iloc[idx, 0] + '.h5'
    file_path = os.path.join(data_path, file_name)
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    label = df.iloc[idx, 1:].values.astype('float32')
    data = load_hdf5_file(file_path).to(device)
    
    # 使用 DeviatTokenizer 生成结果
    variance = deviat_tokenizer(data)

    variance_min = variance.min(dim=0, keepdim=True)[0]
    variance_max = variance.max(dim=0, keepdim=True)[0]
    normalized_variance = (variance - variance_min) / (variance_max - variance_min + 1e-8)

    data_min = data.min(dim=0, keepdim=True)[0]
    data_max = data.max(dim=0, keepdim=True)[0]
    normalized_data = (data - data_min) / (data_max - data_min + 1e-8)

    combined_data = torch.cat([normalized_data, normalized_variance], dim=1)
    return combined_data, torch.tensor(label, dtype=torch.float32).to(device)


# 加载 DeviatTokenizer 的预训练参数
tokenizer_checkpoint_path = os.path.expanduser('~/HOME/ECGLM/Parameters/best_TokenizedMamba2_model_split_100.pth')
deviat_tokenizer = DeviatTokenizer().to(device)
tokenizer_checkpoint = torch.load(tokenizer_checkpoint_path, map_location=device)
deviat_tokenizer.load_state_dict(load_state_dict_without_module_prefix(tokenizer_checkpoint['devia_state_dict']))
deviat_tokenizer.eval()


class ECGDataset(Dataset):
    def __init__(self, df, data_path, normecg, deviat_tokenizer):
        self.df = df
        self.data_path = data_path
        self.normecg = torch.tensor(normecg, dtype=torch.float32).to(device)
        self.deviat_tokenizer = deviat_tokenizer
        self.data_cache = [None] * len(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.data_cache[idx] is None:
            self.data_cache[idx] = load_and_process_data(idx, self.df, self.data_path, self.normecg, self.deviat_tokenizer)
        return self.data_cache[idx]

def save_results_to_csv(results, model_name):
    file_path = '~/HOME/ECGLM//Results/valResult.csv'
    columns = ["Epoch", "Train Loss", "Val Loss", "Train Acc", "Train F1", "Train Precision", "Train Recall", "Train AUC", "Val Acc", "Val F1", "Val Precision", "Val Recall", "Val AUC", "Test Acc", "Test F1", "Test Precision", "Test Recall", "Test AUC"]
    
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(file_path)
    
    for result in results:
        row = dict(zip(columns, result))
        df = df.append(row, ignore_index=True)
    
    df.to_csv(file_path, index=False)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, model_name):
    results = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for data, labels in train_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
                train_labels.append(labels.detach().cpu().numpy())
                train_acc = accuracy_score(np.concatenate(train_labels).round(), np.concatenate(train_preds).round())
                train_f1 = f1_score(np.concatenate(train_labels).round(), np.concatenate(train_preds).round(), average='macro', zero_division=1)
                pbar.set_postfix({'train_loss': train_loss / (pbar.n + 1), 'train_acc': train_acc, 'train_f1': train_f1})
                pbar.update(1)

        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)
        train_acc = accuracy_score(train_labels.round(), train_preds.round())
        train_f1 = f1_score(train_labels.round(), train_preds.round(), average='macro', zero_division=1)
        train_precision = precision_score(train_labels.round(), train_preds.round(), average='macro', zero_division=1)
        train_recall = recall_score(train_labels.round(), train_preds.round(), average='macro', zero_division=1)
        try:
            train_auc = roc_auc_score(train_labels, train_preds, average='macro', multi_class='ovo')
        except ValueError:
            train_auc = 0.0

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc='Validation', unit='batch') as pbar:
                for data, labels in val_loader:
                    data = data.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                    val_labels.append(labels.cpu().numpy())
                    val_acc = accuracy_score(np.concatenate(val_labels).round(), np.concatenate(val_preds).round())
                    val_f1 = f1_score(np.concatenate(val_labels).round(), np.concatenate(val_preds).round(), average='macro', zero_division=1)
                    pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1), 'val_acc': val_acc, 'val_f1': val_f1})
                    pbar.update(1)

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc = accuracy_score(val_labels.round(), val_preds.round())
        val_f1 = f1_score(val_labels.round(), val_preds.round(), average='macro', zero_division=1)
        val_precision = precision_score(val_labels.round(), val_preds.round(), average='macro', zero_division=1)
        val_recall = recall_score(val_labels.round(), val_preds.round(), average='macro', zero_division=1)
        try:
            val_auc = roc_auc_score(val_labels, val_preds, average='macro', multi_class='ovo')
        except ValueError:
            val_auc = 0.0

        results.append([epoch+1, train_loss/len(train_loader), val_loss/len(val_loader), train_acc, train_f1, train_precision, train_recall, train_auc, val_acc, val_f1, val_precision, val_recall, val_auc, None, None, None, None, None])
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Train Acc: {train_acc}, Train F1: {train_f1}, Val Acc: {val_acc}, Val F1: {val_f1}')
    
    return results

dataset = ECGDataset(df, data_path, average_norm_ecg_signal_10, deviat_tokenizer)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print("Preloading data...")
# with ThreadPoolExecutor(max_workers=40) as executor:
    # list(tqdm(executor.map(lambda idx: load_and_process_data(idx, df, data_path, average_norm_ecg_signal_10), range(len(dataset))), total=len(dataset), desc='Data Loading'))

kf = KFold(n_splits=10)
fold_results = []

model_name = "AbnormalTokenizer"

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = TransformerClassifier(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    fold_results.extend(train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, model_name))

    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Validation Fold {fold+1}', unit='batch') as pbar:
            for data, labels in val_loader:
                data = data.to(device)
                outputs = model(data)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                pbar.update(1)

    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)
    fold_acc = accuracy_score(val_labels.round(), val_preds.round())
    fold_f1 = f1_score(val_labels.round(), val_preds.round(), average='macro', zero_division=1)
    fold_precision = precision_score(val_labels.round(), val_preds.round(), average='macro', zero_division=1)
    fold_recall = recall_score(val_labels.round(), val_preds.round(), average='macro', zero_division=1)
    try:
        fold_auc = roc_auc_score(val_labels, val_preds, average='macro', multi_class='ovo')
    except ValueError:
        fold_auc = 0.0

    fold_results.append([f"Fold {fold+1}", None, None, None, None, fold_acc, fold_f1, fold_precision, fold_recall, fold_auc])
    print(f'Fold {fold+1} Accuracy: {fold_acc}, F1 Score: {fold_f1}, Precision: {fold_precision}, Recall: {fold_recall}, AUC: {fold_auc}')

avg_acc = np.mean([result[5] for result in fold_results if result[5] is not None])
avg_f1 = np.mean([result[6] for result in fold_results if result[6] is not None])
avg_precision = np.mean([result[7] for result in fold_results if result[7] is not None])
avg_recall = np.mean([result[8] for result in fold_results if result[8] is not None])
avg_auc = np.mean([result[9] for result in fold_results if result[9] is not None])

print(f'Average Accuracy: {avg_acc}, Average F1 Score: {avg_f1}, Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average AUC: {avg_auc}')

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    with tqdm(total=len(test_loader), desc='Test Evaluation', unit='batch') as pbar:
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            test_preds.append(torch.sigmoid(outputs).cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            pbar.update(1)

test_preds = np.concatenate(test_preds)
test_labels = np.concatenate(test_labels)
test_acc = accuracy_score(test_labels.round(), test_preds.round())
test_f1 = f1_score(test_labels.round(), test_preds.round(), average='macro', zero_division=1)
test_precision = precision_score(test_labels.round(), test_preds.round(), average='macro', zero_division=1)
test_recall = recall_score(test_labels.round(), test_preds.round(), average='macro', zero_division=1)
try:
    test_auc = roc_auc_score(test_labels, test_preds, average='macro', multi_class='ovo')
except ValueError:
    test_auc = 0.0

print(f'Test Accuracy: {test_acc}, Test F1 Score: {test_f1}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test AUC: {test_auc}')
fold_results.append(["Test", None, None, None, None, test_acc, test_f1, test_precision, test_recall, test_auc])

save_results_to_csv(fold_results, model_name)
print("Results saved to CSV.")
