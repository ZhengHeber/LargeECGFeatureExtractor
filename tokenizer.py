import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import sys
from mamba_ssm import Mamba2
import torch.fft


class NullTokenizer:
    """
    This class provides methods to tokenize ECG data by padding and reshaping it.
    """
    def __init__(self, chunk_size=250, num_chunks=256, target_length=320):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.target_length = target_length

    def tokenize(self, data):
        b, l, c = data.shape
        data = rearrange(data, 'b l c -> b (l c)')

        # Pad data with zeros to reach 64000
        padding_length = 64000 - data.shape[1]
        data = torch.cat([data, torch.zeros(b, padding_length, device=data.device)], dim=1)

        # Reshape to (b, num_chunks, chunk_size)
        data = data.reshape(b, self.num_chunks, self.chunk_size).transpose(1, 2)

        # Add 70 zeros along the second dimension
        addLen = self.target_length - self.chunk_size
        zeros_to_add = torch.zeros(b, addLen, self.num_chunks, device=data.device)
        data = torch.cat([data, zeros_to_add], dim=1)

        return data

class ECGMetaTokenizer(nn.Module):
    def __init__(self, chunk_size=250, num_chunks=256, target_length=270):
        super(ECGMetaTokenizer, self).__init__()
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.target_length = target_length
        self.mask = nn.Parameter(torch.ones((self.target_length - self.chunk_size), 1))  # 添加的掩码参数

    def _pad_data_block(self, data_block):
        """
        Pads the data block to the target length and appends meta features.
        """
        pad_size = self.target_length - data_block.shape[0]
        meta_features = self.metaFeature(data_block) * self.mask  # 应用掩码

        # 计算 meta_features 的均值和标准差
        mean = meta_features.mean(dim=0, keepdim=True)
        std = meta_features.std(dim=0, keepdim=True)

        # 对 meta_features 进行归一化
        meta_features = (meta_features - mean) / std

        return torch.cat([data_block, meta_features.to(data_block.device)], dim=0) 

    def metaFeature(self, data_block):
        """
        Extracts meta features from the data block.
        """
        device = data_block.device
        mean = torch.mean(data_block, dim=0, keepdim=True)
        median = torch.median(data_block, dim=0, keepdim=True).values
        max_val = torch.max(data_block, dim=0, keepdim=True).values
        min_val = torch.min(data_block, dim=0, keepdim=True).values
        std = torch.std(data_block, dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(data_block**2, dim=0, keepdim=True))
        zero_crossings = torch.sum((data_block[:-1] * data_block[1:]) < 0, dim=0, keepdim=True)
        mad = torch.mean(torch.abs(data_block - mean), dim=0, keepdim=True)
        rise_time = torch.argmax(data_block, dim=0, keepdim=True).float() / data_block.shape[0]
        fall_time = torch.argmax(data_block.flip(dims=[0]), dim=0, keepdim=True).float() / data_block.shape[0]
        amplitude = (max_val - min_val) / 2
        sample_entropy = -torch.sum(F.softmax(data_block, dim=0) * F.log_softmax(data_block, dim=0), dim=0, keepdim=True)
        mean_amplitude_change_rate = torch.mean(torch.abs(data_block[1:] - data_block[:-1]), dim=0, keepdim=True)
        mssd = torch.mean((data_block[1:] - data_block[:-1])**2, dim=0, keepdim=True)
        linear_trend_slope, linear_trend_intercept = self.LTCalc(data_block)
        energy = torch.sum(data_block**2, dim=0, keepdim=True)
        iqr = torch.quantile(data_block, 0.75, dim=0, keepdim=True) - torch.quantile(data_block, 0.25, dim=0, keepdim=True)
        autocorrelation = torch.mean((data_block - mean) * (data_block - mean), dim=0, keepdim=True)
        log_energy_entropy = -torch.sum(data_block**2 * torch.log(data_block**2 + 1e-9), dim=0, keepdim=True)
        features = torch.cat([
            mean, median, max_val, min_val, std, rms, zero_crossings,
            mad, rise_time, fall_time, amplitude, sample_entropy,
            mean_amplitude_change_rate, mssd, linear_trend_slope, linear_trend_intercept,
            energy, iqr, autocorrelation, log_energy_entropy
        ], dim=0)
        return features.to(device)

    def LTCalc(self, data_block):
        """
        Calculates the linear trend slope and intercept for the data block.
        """
        n = data_block.shape[0]
        x = torch.arange(n, dtype=torch.float32, device=data_block.device).reshape(-1, 1)
        y = data_block
        sum_x = torch.sum(x)
        sum_y = torch.sum(y, dim=0, keepdim=True)
        sum_xy = torch.sum(x * y, dim=0, keepdim=True)
        sum_x2 = torch.sum(x**2)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    def tokenize(self, data):
        """
        Tokenizes the data by reshaping and padding it, and extracting meta features.
        """
        b, l, c = data.shape
        data = rearrange(data, 'b l c -> b (l c)')
        padding_length = 64000 - data.shape[1]
        data = torch.cat([data, torch.zeros(b, padding_length, device=data.device)], dim=1)
        data = data.reshape(b, self.num_chunks, self.chunk_size).transpose(1, 2)
        padded_data = torch.stack([self._pad_data_block(data[i]) for i in range(b)], dim=0)
        return padded_data.to(data.device)

    def forward(self, data):
        _ = self.mask * 1.0
        return data

class ECGSpaceTokenizer(nn.Module):
    def __init__(self, chunk_size=270, num_chunks=256, target_length=320, raw_length=250):
        super(ECGSpaceTokenizer, self).__init__()
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.target_length = target_length
        self.raw_length = raw_length
        self.sparse_feature_matrix = nn.Parameter(torch.empty(240, 250, 11))
        nn.init.xavier_uniform_(self.sparse_feature_matrix)

    def SparseSpatialFeature(self, spatial_matrix):
        """
        Generates sparse spatial features for the data block.
        """
        b, num_matrix, chunk_size, c = spatial_matrix.shape
        multi_lead_splines = []

        for i in range(240):
            n_matrix = i % 20
            n_lead = i // 20
            spatial_chunk = spatial_matrix[:, n_matrix, :, :].to(self.sparse_feature_matrix.device)
            exclude_spatial_matrix = torch.cat((spatial_chunk[:, :, :n_lead], spatial_chunk[:, :, n_lead+1:]), dim=2)
            sparse_matrix = self.sparse_feature_matrix[i].unsqueeze(0).expand(b, -1, -1)  # 扩展批次维度
            multi_lead_spline = torch.mul(exclude_spatial_matrix, sparse_matrix).sum(dim=-1)
            multi_lead_splines.append(multi_lead_spline)

        for i in range(240, self.num_chunks):
            multi_lead_splines.append(torch.zeros(b, self.raw_length).to(self.sparse_feature_matrix.device))

        multi_lead_splines = torch.stack(multi_lead_splines, dim=1)
        multi_lead_splines = F.interpolate(multi_lead_splines, size=(50,), mode='linear', align_corners=True)
        multi_lead_splines = multi_lead_splines.transpose(1, 2)

        # Normalize the spacial features along the first dimension
        spacial_features_mean = multi_lead_splines.mean(dim=1, keepdim=True)
        spacial_features_std = multi_lead_splines.std(dim=1, keepdim=True)
        multi_lead_splines = (multi_lead_splines - spacial_features_mean) / (spacial_features_std + 1e-8)

        return multi_lead_splines

    def _pad_data_block(self, data_block, spacial_features):
        """
        Pads the data block with sparse spatial features.
        """
        pad_size = self.target_length - self.chunk_size
        spacial_features = spacial_features[:, :pad_size, :].to(data_block.device)  # 确保在 GPU 上
        return torch.cat([data_block, spacial_features], dim=1)

    def tokenize(self, data, data_time_padded):
        """
        Tokenizes the data by reshaping and padding it, and extracting sparse spatial features.
        """
        b, l, c = data.shape
        num_matrix = l // self.raw_length
        spatial_matrix = rearrange(data, 'b (num_matrix chunk_size) c -> b num_matrix chunk_size c', chunk_size=self.raw_length)
        spacial_features = self.SparseSpatialFeature(spatial_matrix)
        padded_data = self._pad_data_block(data_time_padded, spacial_features)
        return padded_data

    def forward(self, data):
        _ = self.sparse_feature_matrix * 1.0
        return data
    
    
class CrossTokenizer(nn.Module):
    def __init__(self, chunk_size=270, num_chunks=256, raw_length=5000):
        super(CrossTokenizer, self).__init__()
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.raw_length = raw_length
        self.sparse_feature_matrix = nn.Parameter(torch.empty(100, 12, 5000, 11))
        self.feature_selection_matrix = nn.Parameter(torch.empty(100, 12, 11, 5000))
        nn.init.xavier_uniform_(self.sparse_feature_matrix)
        nn.init.xavier_uniform_(self.feature_selection_matrix)

        # Check for NaNs after initialization
        if torch.isnan(self.sparse_feature_matrix).any():
            raise ValueError("NaN detected in sparse_feature_matrix after initialization.")
        if torch.isnan(self.feature_selection_matrix).any():
            raise ValueError("NaN detected in feature_selection_matrix after initialization.")

        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(5000)  # Assuming batch normalization over the time dimension

    def SparseSpatialFeature(self, ecg_data):
        ecg_data = ecg_data.permute(0, 2, 1)  # Shape: (b, 12, 5000)
        b, num_leads, length = ecg_data.shape
        assert length == self.raw_length, "Input data length must match raw_length."
        multi_lead_splines = []

        for i in range(num_leads):
            ecg_lead = ecg_data[:, i, :].unsqueeze(-1)  # Shape: (b, 5000, 1)
            exclude_lead_matrix = torch.cat((ecg_data[:, :i, :], ecg_data[:, i+1:, :]), dim=1)  # Shape: (b, 11, 5000)
            exclude_lead_matrix = exclude_lead_matrix.permute(0, 2, 1)  # Shape: (b, 5000, 11)
            
            sparse_matrix = self.sparse_feature_matrix[:b, i]  # Shape: (b, 5000, 11)
            if torch.isnan(sparse_matrix).any():
                raise ValueError(f"NaN detected in sparse_matrix for lead {i} during SparseSpatialFeature.")

            sparse_features = torch.mul(exclude_lead_matrix, sparse_matrix)  # Shape: (b, 5000, 11)
            if torch.isnan(sparse_features).any():
                raise ValueError(f"NaN detected in sparse_features for lead {i} during SparseSpatialFeature.")

            # sparse_features = self.batch_norm(sparse_features.reshape(-1, 5000)).reshape(b, 5000, 11)
            # sparse_features = self.relu(sparse_features)

            feature_selection = self.feature_selection_matrix[:b, i]  # Shape: (b, 11, 5000)
            if torch.isnan(feature_selection).any():
                raise ValueError(f"NaN detected in feature_selection for lead {i} during SparseSpatialFeature.")

            selected_features = torch.einsum('bfe,bef->bf', sparse_features, feature_selection)  # Shape: (b, 5000)
            if torch.isnan(selected_features).any():
                raise ValueError(f"NaN detected in selected_features for lead {i} during SparseSpatialFeature.")

            multi_lead_splines.append(selected_features)

        multi_lead_splines = torch.stack(multi_lead_splines, dim=2)  # Shape: (b, 5000, 12)

        # Apply Batch Normalization
        multi_lead_splines = self.batch_norm(multi_lead_splines.reshape(-1, 5000)).reshape(b, 5000, 12)

        # Apply ReLU activation
        multi_lead_splines = self.relu(multi_lead_splines)

        return multi_lead_splines

    def _pad_data_block(self, data_block, spatial_features):
        spatial_features = spatial_features.to(data_block.device)  # Ensure on GPU
        return spatial_features

    def tokenize(self, data):
        b, l, c = data.shape
        assert c == 12, "Input data must have 12 leads."
        
        if torch.isnan(data).any():
            raise ValueError("NaN detected in input data.")

        spatial_features = self.SparseSpatialFeature(data)
        padded_data = self._pad_data_block(data, spatial_features)
        
        if torch.isnan(padded_data).any():
            raise ValueError("NaN detected in padded_data after tokenization.")
        
        return padded_data

    def forward(self, data):
        return self.tokenize(data)
    


class DeviatTokenizer(nn.Module):
    def __init__(self, d_model=256, d_state=64, d_conv=4, dropout_rate=0.1, codebook_shape=(500, 5000, 12)):
        super(DeviatTokenizer, self).__init__()
        self.codebook_shape = codebook_shape
        self.hash_table = nn.Parameter(torch.rand(codebook_shape[0], 12 * 2))  # 每个codebook的哈希值
        self.mamba2_block = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout_rate=dropout_rate)

    def fft_hash(self, data):
        fft_features = torch.fft.rfft(data, dim=1)  # Shape: (batch_size, frequency_bins, 12)
        magnitude = torch.abs(fft_features)  # 取幅值
        phase = torch.angle(fft_features)  # 取相位

        # 将幅值和相位拼接起来
        combined_features = torch.cat([magnitude, phase], dim=1)  # Shape: (batch_size, frequency_bins * 2, 12)
        fft_mean = torch.mean(combined_features, dim=1)  # Shape: (batch_size, 12 * 2)

        # 简单哈希计算: 比较哈希表中的每一个codebook的均值和当前数据的均值，找出最相似的
        hash_indices = torch.argmin(torch.norm(self.hash_table.unsqueeze(0) - fft_mean.unsqueeze(1), dim=-1), dim=1)
        return hash_indices

    def forward(self, data, codebook):
        hash_indices = self.fft_hash(data)

        closest_ecg_mean = codebook[hash_indices]

        # 计算方差并标准化
        variance = (data - closest_ecg_mean) ** 2
        variance_min = variance.min(dim=0, keepdim=True)[0]
        variance_max = variance.max(dim=0, keepdim=True)[0]
        normalized_variance = (variance - variance_min) / (variance_max - variance_min + 1e-8)

        # 用Mamba2Block进行拟合
        output = self.mamba2_block(data, normalized_variance)

        return output