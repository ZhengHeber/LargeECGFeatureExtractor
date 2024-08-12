import h5py
import os

def list_hdf5_file_contents(file_path):
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    # 打开 HDF5 文件
    with h5py.File(file_path, 'r') as f:
        print("Listing HDF5 file contents:")
        f.visititems(print_structure)

if __name__ == "__main__":
    # file_path = os.path.expanduser('/home/zhenghb/HOME/ECGLM/Validations/Google/home/yshi7084/Datasets/ptbxl_pred/p_10087_hr.hdf5')
    # file_path = os.path.expanduser('~/HOME/dinov/output/118113003451/1181130034511.hdf5')
    file_path = os.path.expanduser('~/HOME/dinov/output/input2/00001_hr_processed.hdf5')
    list_hdf5_file_contents(file_path)
