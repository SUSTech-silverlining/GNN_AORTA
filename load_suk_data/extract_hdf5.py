import h5py
import argparse
from tqdm import tqdm

def extract_samples(input_file, output_file, num_samples):
    """
    從一個 HDF5 檔案中提取指定數量的樣本到一個新的 HDF5 檔案中。

    Args:
        input_file (str): 來源 HDF5 檔案的路徑。
        output_file (str): 目標 HDF5 檔案的路徑。
        num_samples (int): 要提取的樣本數量。
    """
    print(f"Opening source file: {input_file}")
    
    try:
        with h5py.File(input_file, 'r') as f_in:
            # 獲取所有樣本的 ID (keys)
            sample_ids = sorted(list(f_in.keys()))
            
            if len(sample_ids) < num_samples:
                print(f"Warning: Requested {num_samples} samples, but only {len(sample_ids)} found.")
                num_samples = len(sample_ids)

            # 選取前 num_samples 個樣本
            samples_to_copy = sample_ids[:num_samples]
            print(f"Found {len(sample_ids)} total samples. Copying the first {len(samples_to_copy)}...")

            with h5py.File(output_file, 'w') as f_out:
                for sample_id in tqdm(samples_to_copy, desc="Copying samples"):
                    # 將每個樣本的整個群組 (group) 複製到新檔案
                    f_in.copy(sample_id, f_out)
            
            print(f"\nSuccessfully created new HDF5 file at: {output_file}")
            print(f"It contains {len(samples_to_copy)} samples.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract a subset of samples from an HDF5 file.")
    parser.add_argument('--input', required=True, help="Path to the source HDF5 file.")
    parser.add_argument('--output', required=True, help="Path for the new, smaller HDF5 file.")
    parser.add_argument('--num', type=int, default=10, help="Number of samples to extract.")
    
    args = parser.parse_args()
    
    extract_samples(args.input, args.output, args.num)
