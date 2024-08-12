import os
import h5py
import wfdb
import numpy as np
from scipy.io import loadmat
import asyncio
from concurrent.futures import ProcessPoolExecutor
import time

def convert_dat_to_hdf5(dat_file, output_file):
    try:
        record = wfdb.rdrecord(dat_file.replace('.dat', ''), sampfrom=0, physical=False)
        signal = record.d_signal

        # if signal.shape[0] != 5000 or signal.shape[1] != 12:
        #     raise ValueError(f"Signal shape is {signal.shape}, expected [5000, 12]")

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('signal', data=signal)

        # print(f"Converted {dat_file} to {output_file}")


    except Exception as e:
        print(f"Error processing {dat_file}: {e}")

def convert_mat_to_hdf5(mat_file, output_file):
    try:
        mat_data = loadmat(mat_file)
        signal = mat_data['val']

        # if signal.shape[0] != 5000 or signal.shape[1] != 12:
            # raise ValueError(f"Signal shape is {signal.shape}, expected [5000, 12]")

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('signal', data=signal)
        print(f"Converted {mat_file} to {output_file}")

    except Exception as e:
        print(f"Error processing {mat_file}: {e}")

def process_file(input_file, output_file, convert_function):
    convert_function(input_file, output_file)

async def process_directory(input_root, output_root, file_extension, convert_function, max_workers=4):
    if not os.path.exists(input_root):
        print(f"Input directory {input_root} does not exist")
        return

    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory {output_root}")

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for root, _, files in os.walk(input_root):
            for file in files:
                if file.endswith(file_extension):
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(output_root, file.replace(file_extension, '.hdf5'))
                    tasks.append(loop.run_in_executor(executor, process_file, input_file, output_file, convert_function))

        await asyncio.gather(*tasks)

if __name__ == "__main__":
    # mimic_input_root = 'F:\\ecg\\mimic-iv-ecg\\mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0\\files'
    # output_root = 'E:\\ecg\\ECG_HDF5\\Mimic'

    # wfdblead = 'F:\\ecg\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0 (1)\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\WFDBRecords'
    # output_root = 'E:\\ecg\\ECG_HDF5\\wfdblead'
    ptbxl = 'F:\\ecg\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 (1)\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1\\records500'
    output_root = 'E:\\ecg\\ECG_HDF5\\ptbxl'

    # Adjust the number of workers based on your system's CPU cores
    asyncio.run(process_directory(ptbxl, output_root, '.dat', convert_dat_to_hdf5, max_workers=48))
    # asyncio.run(process_directory(wfdblead, output_root, '.mat', convert_mat_to_hdf5, max_workers=48))
