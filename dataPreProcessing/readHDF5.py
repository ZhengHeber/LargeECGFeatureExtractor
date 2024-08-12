import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def display_ecg_lead(data, title, sampling_rate, duration=10):
    """
    Display a single ECG lead.
    """
    time = np.arange(0, len(data)) / sampling_rate
    plt.plot(time[:sampling_rate * duration], data[:sampling_rate * duration])
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

def display_hdf5_file(file_path):
    # 打开 HDF5 文件
    with h5py.File(file_path, 'r') as f:
        # 读取数据集
        signal = f['signal'][:]
        
        # 通过数据形状确定采样率和通道数
        sampling_rate = signal.shape[0] // 10  # 假设数据长度为10秒
        num_channels = signal.shape[1]
        
        # 显示属性信息
        print(f'Sampling Rate: {sampling_rate} Hz')
        print(f'Number of Channels: {num_channels}')
        print(f'Signal Shape: {signal.shape}')
        
        # 导联名称
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if num_channels < 12:
            leads = leads[:num_channels]
        
        # 绘制 ECG 信号
        plt.figure(figsize=(15, 10))
        for i, lead in enumerate(leads):
            plt.subplot(4, 3, i + 1)
            display_ecg_lead(signal[:, i], lead, sampling_rate)
        
        plt.tight_layout()
        plt.show()
        plt.savefig('/home/zhenghb/HOME/ECGLLM/ECG/Fig/SampleECG.png')

if __name__ == "__main__":
    # file_path = 'E:\\ecg\\ECG_HDF5\\Mimic\\p1000\\p10000032\\s40689238\\40689238.hdf5'
    # file_path = 'E:\\ecg\\ECG_HDF5\\Head12l\\01\\010\\JS00001.hdf5'
    # file_path = 'E:\\ecg\\ECG_HDF5\\Ptb\\records100\\00000\\00001_lr.hdf5'
    file_path = os.path.expanduser('/home/zhenghb/HOME/ECGLLM/ECG/SampleHDF5/40895702.hdf5')
    display_hdf5_file(file_path)
