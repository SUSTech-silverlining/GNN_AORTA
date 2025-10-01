import h5py
import argparse

def inspect_file(filepath):
    """
    打開一個 HDF5 檔案，打印出第一個樣本的所有數據集名稱 (keys)。
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if not f.keys():
                print(f"Error: The HDF5 file '{filepath}' is empty or has no top-level groups.")
                return

            # 獲取第一個樣本的 ID
            first_sample_id = sorted(list(f.keys()))[0]
            print(f"--- Inspecting sample '{first_sample_id}' in '{filepath}' ---")
            
            sample_group = f[first_sample_id]
            
            print("Available dataset keys for this sample are:")
            # 打印出這個樣本中所有數據集的名稱
            for key in sample_group.keys():
                print(f"- {key}")

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect the keys of the first sample in an HDF5 file.")
    parser.add_argument('--file', required=True, help="Path to the HDF5 file to inspect.")
    
    args = parser.parse_args()
    inspect_file(args.file)
