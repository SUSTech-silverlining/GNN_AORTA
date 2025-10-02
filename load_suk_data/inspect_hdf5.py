import h5py
import argparse

def inspect_file(filepath):
    """
    Open an HDF5 file and print all dataset names (keys) of the first sample.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if not f.keys():
                print(f"Error: The HDF5 file '{filepath}' is empty or has no top-level groups.")
                return

            # Get the ID of the first sample
            first_sample_id = sorted(list(f.keys()))[0]
            print(f"--- Inspecting sample '{first_sample_id}' in '{filepath}' ---")
            
            sample_group = f[first_sample_id]
            
            print("Available dataset keys for this sample are:")
            # Print all dataset names in this sample
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
