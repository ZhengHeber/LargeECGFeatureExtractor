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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Transformer model
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
        dummy_input = torch.zeros(1, 5000, 12)
        x = dummy_input.to(next(self.parameters()).device)
        x = x.transpose(1, 2) 
        x = self.initial_conv(x)
        x = x.transpose(2, 1) 
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)  
        x = self.final_conv(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.initial_conv(x)
        x = x.transpose(2, 1) 
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)  
        x = self.final_conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Paths
label_file_path = os.path.expanduser('~/HOME/ECGLM/Validations/ptbxl_ecg_data_label.csv')
# data_path = os.path.expanduser('~/HOME/ECGLM/Validations/ptbxl')
data_path = os.path.expanduser('~/HOME/dinov/output/ptbxl')

# Hyperparameters
num_epochs = 8
batch_size = 32
learning_rate = 0.0001
num_classes = 4  # CD, HYP, STTC, MI

# Load labels
print("Loading labels...")
df = pd.read_csv(label_file_path)
df = df[['filename_hr', 'CD', 'HYP', 'STTC', 'MI']]

# Function to load HDF5 file
def load_hdf5_file(file_path):
    """
    Loads HDF5 file and returns the dataset.
    """
    with h5py.File(file_path, 'r') as f:
        # data = f['signal'][:]  # Assuming dataset is stored in 'signal' dataset
        data = f['signal'][:]
    return torch.tensor(data, dtype=torch.float32)

# Function to load data and calculate variance
def load_and_process_data(idx, df, data_path):
    file_name = df.iloc[idx, 0].replace('/ptbxl/', '')
    file_path = os.path.join(data_path, file_name)
    file_path = os.path.abspath(file_path)  # Ensure absolute path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    label = df.iloc[idx, 1:].values.astype('float32')
    data = load_hdf5_file(file_path).to(device)

    data_min = data.min(dim=0, keepdim=True)[0]
    data_max = data.max(dim=0, keepdim=True)[0]
    normalized_data = (data - data_min) / (data_max - data_min + 1e-8)

    return normalized_data, torch.tensor(label, dtype=torch.float32).to(device)

class ECGDataset(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = data_path
        self.data_cache = [None] * len(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.data_cache[idx] is None:
            self.data_cache[idx] = load_and_process_data(idx, self.df, self.data_path)
        return self.data_cache[idx]

# Save results to CSV
def save_results_to_csv(results, model_name):
    file_path = '~/HOME/ECGLM//Results/valResult.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("Model, Epoch, Train Loss, Val Loss, Train Acc, Train F1, Train Precision, Train Recall, Train AUC, "
                    "Val Acc, Val F1, Val Precision, Val Recall, Val AUC, "
                    "Test Acc, Test F1, Test Precision, Test Recall, Test AUC\n")
    
    with open(file_path, 'a') as f:
        for result in results:
            f.write(f"{model_name}, " + ", ".join(map(str, result)) + "\n")

# Training and validation function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, model_name):
    results = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for data, labels in train_loader:
                # data = data.permute(0, 2, 1)
                outputs = model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
                train_labels.append(labels.detach().cpu().numpy())
                pbar.update(1)

        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)
        train_acc = accuracy_score(train_labels.round(), train_preds.round())
        train_f1 = f1_score(train_labels.round(), train_preds.round(), average='macro')
        train_precision = precision_score(train_labels.round(), train_preds.round(), average='macro')
        train_recall = recall_score(train_labels.round(), train_preds.round(), average='macro')
        train_auc = roc_auc_score(train_labels, train_preds, average='macro', multi_class='ovo')

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc='Validation', unit='batch') as pbar:
                for data, labels in val_loader:
                    # data = data.permute(0, 2, 1)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                    val_labels.append(labels.cpu().numpy())
                    pbar.update(1)

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc = accuracy_score(val_labels.round(), val_preds.round())
        val_f1 = f1_score(val_labels.round(), val_preds.round(), average='macro')
        val_precision = precision_score(val_labels.round(), val_preds.round(), average='macro')
        val_recall = recall_score(val_labels.round(), val_preds.round(), average='macro')
        val_auc = roc_auc_score(val_labels, val_preds, average='macro', multi_class='ovo')

        results.append([epoch+1, train_loss/len(train_loader), val_loss/len(val_loader), 
                        train_acc, train_f1, train_precision, train_recall, train_auc,
                        val_acc, val_f1, val_precision, val_recall, val_auc,
                        None, None, None, None, None])
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, '
              f'Train Acc: {train_acc}, Train F1: {train_f1}, Train Precision: {train_precision}, Train Recall: {train_recall}, Train AUC: {train_auc}, '
              f'Val Acc: {val_acc}, Val F1: {val_f1}, Val Precision: {val_precision}, Val Recall: {val_recall}, Val AUC: {val_auc}')
    
    return results

# Split dataset into training and testing
dataset = ECGDataset(df, data_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# K-Fold Cross Validation
kf = KFold(n_splits=10)
fold_results = []

model_name = "TransformerClassifier"

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

    # Evaluation on validation set
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Validation Fold {fold+1}', unit='batch') as pbar:
            for data, labels in val_loader:
                # data = data.permute(0, 2, 1)
                data = data.to(device)
                outputs = model(data)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                pbar.update(1)

    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)
    fold_acc = accuracy_score(val_labels.round(), val_preds.round())
    fold_f1 = f1_score(val_labels.round(), val_preds.round(), average='macro')
    fold_precision = precision_score(val_labels.round(), val_preds.round(), average='macro')
    fold_recall = recall_score(val_labels.round(), val_preds.round(), average='macro')
    fold_auc = roc_auc_score(val_labels, val_preds, average='macro', multi_class='ovo')

    fold_results.append([f"Fold {fold+1}", None, None, None, None, fold_acc, fold_f1, fold_precision, fold_recall, fold_auc])
    
    print(f'Fold {fold+1} Accuracy: {fold_acc}, F1 Score: {fold_f1}, Precision: {fold_precision}, Recall: {fold_recall}, AUC: {fold_auc}')

# Average results over all folds
avg_acc = np.mean([result[5] for result in fold_results if result[5] is not None])
avg_f1 = np.mean([result[6] for result in fold_results if result[6] is not None])
avg_precision = np.mean([result[7] for result in fold_results if result[7] is not None])
avg_recall = np.mean([result[8] for result in fold_results if result[8] is not None])
avg_auc = np.mean([result[9] for result in fold_results if result[9] is not None])

print(f'Average Accuracy: {avg_acc}, Average F1 Score: {avg_f1}, Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average AUC: {avg_auc}')

# Final evaluation on test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    with tqdm(total=len(test_loader), desc='Test Evaluation', unit='batch') as pbar:
        for data, labels in test_loader:
            # data = data.permute(0, 2, 1)
            outputs = model(data)
            test_preds.append(torch.sigmoid(outputs).cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            pbar.update(1)

test_preds = np.concatenate(test_preds)
test_labels = np.concatenate(test_labels)
test_acc = accuracy_score(test_labels.round(), test_preds.round())
test_f1 = f1_score(test_labels.round(), test_preds.round(), average='macro')
test_precision = precision_score(test_labels.round(), test_preds.round(), average='macro')
test_recall = recall_score(test_labels.round(), test_preds.round(), average='macro')
test_auc = roc_auc_score(test_labels, test_preds, average='macro', multi_class='ovo')

print(f'Test Accuracy: {test_acc}, Test F1 Score: {test_f1}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test AUC: {test_auc}')
fold_results.append(["Test", None, None, None, None, test_acc, test_f1, test_precision, test_recall, test_auc])

save_results_to_csv(fold_results, model_name)
print("Results saved to CSV.")
