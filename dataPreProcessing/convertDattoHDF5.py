import wfdb
import h5py
import os
import numpy as np
from scipy.io import loadmat

def convert_dat_to_hdf5(dat_file, output_file):
    try:
        # 读取.dat文件中的ECG信号
        # print(f"Reading {dat_file}")
        record = wfdb.rdrecord(dat_file.replace('.dat', ''), sampfrom=0, physical=False)
        signal = record.d_signal

        # 保证信号数据为[5000, 12]
        # print(f"Signal shape: {signal.shape}")
        if signal.shape[0] != 5000 or signal.shape[1] != 12:
            raise ValueError(f"Signal shape is {signal.shape}, expected [5000, 12]")

        # 将信号保存为.hdf5格式
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('signal', data=signal)
        print(f"Converted {dat_file} to {output_file}")

    except Exception as e:
        print(f"Error processing {dat_file}: {e}")

def convert_mat_to_hdf5(mat_file, output_file):
    try:
        # 读取.mat文件中的ECG数据
        # print(f"Reading {mat_file}")
        mat_data = loadmat(mat_file)
        signal = mat_data['val']

        # 保证信号数据为[5000, 12]
        # print(f"Signal shape: {signal.shape}")
        if signal.shape[0] != 5000 or signal.shape[1] != 12:
            raise ValueError(f"Signal shape is {signal.shape}, expected [5000, 12]")

        # 将信号保存为.hdf5格式
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('signal', data=signal)
        print(f"Converted {mat_file} to {output_file}")

    except Exception as e:
        print(f"Error processing {mat_file}: {e}")

def process_directory(input_root, output_root, file_extension, convert_function):
    if not os.path.exists(input_root):
        print(f"Input directory {input_root} does not exist")
        return

    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory {output_root}")

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith(file_extension):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_root, file.replace(file_extension, '.hdf5'))
                print(f"Processing {input_file}")
                convert_function(input_file, output_file)

if __name__ == "__main__":
    # Uncomment these lines to test with other datasets
    # mimic_input_root = 'E:\\ecg\\mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0\\files'
    # head12l_input_root = 'E:\\ecg\\12head_large_ECG-1.0.0\\WFDBRecords'
    # ptb_input_root = 'E:\\ecg\\ptb-xl-1.0.1'
    # output_root = 'E:\\ecg\\ECG_HDF5\\'

    # process_directory(mimic_input_root, output_root, '.dat', convert_dat_to_hdf5)
    # process_directory(head12l_input_root, output_root, '.mat', convert_mat_to_hdf5)
    # process_directory(ptb_input_root, output_root, '.dat', convert_dat_to_hdf5)

    sampleECG = os.path.expanduser('~/HOME/ECGLLM/ECG/ECGSample')
    output_root = os.path.expanduser('~/HOME/ECGLLM/ECG/SampleHDF5')

    process_directory(sampleECG, output_root, '.dat', convert_dat_to_hdf5)
