import os
import struct
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

INPUT_ROOT = r"d:/forKrishna/ECT_EEG/ECT_EEG"
OUTPUT_512 = r"d:/forKrishna/ECT_EEG/csv_512Hz"
OUTPUT_1000 = r"d:/forKrishna/ECT_EEG/csv_1000Hz"

def extract_data(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()

    # Skip Header (Starts at offset 16)
    start_idx = 16
    
    # Find Footer "NQR" 
    footer_idx = content.find(b'NQR', start_idx)
    
    if footer_idx == -1:
        data_block = content[start_idx:]
    else:
        data_block = content[start_idx:footer_idx]

    num_samples = len(data_block) // 16

    ch1_AC, ch2_AC, ch3_AC, ch4 = [], [], [], []
    ch1_DC, ch2_DC, ch3_DC = [], [], []

    # Unpack binary data
    for i in range(num_samples):
        offset = i * 16
        vals = struct.unpack('<8H', data_block[offset : offset+16])
        
        ch1_AC.append(vals[0])
        ch2_AC.append(vals[1])
        ch3_AC.append(vals[2])
        ch4.append(vals[3]) 
        
        ch1_DC.append(vals[4]) 
        ch2_DC.append(vals[5]) 
        ch3_DC.append(vals[6]) 

    # Reconstruct Full Signal: AC + DC
    final_ch1 = np.array(ch1_AC, dtype=np.float32) + np.array(ch1_DC, dtype=np.float32)
    final_ch2 = np.array(ch2_AC, dtype=np.float32) + np.array(ch2_DC, dtype=np.float32)
    final_ch3 = np.array(ch3_AC, dtype=np.float32) + np.array(ch3_DC, dtype=np.float32)
    final_ch4 = np.array(ch4, dtype=np.float32)
    
    return final_ch1, final_ch2, final_ch3, final_ch4, num_samples


# Define the data loader script content that will be written into each folder
LOADER_SCRIPT_CONTENT = """import os
import pandas as pd
import glob

def load_all_data():
    \"\"\"
    Loads all CSV files in the current folder into a dictionary of DataFrames.
    Returns:
        dict: A dictionary where keys are the filenames (without .csv) and 
              values are pandas DataFrames containing the structured EEG data.
    \"\"\"
    folder = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(folder, "*.csv"))
    
    data_dict = {}
    for f in files:
        name = os.path.basename(f).replace('.csv', '')
        data_dict[name] = pd.read_csv(f)
    return data_dict

def load_data(filename):
    \"\"\"
    Loads a specific CSV file by name.
    Args:
        filename (str): The name of the file (e.g., '2ASZPCRC11004' or '2ASZPCRC11004.csv')
    Returns:
        pd.DataFrame: A pandas DataFrame containing the EEG data.
    \"\"\"
    if not filename.endswith('.csv'):
        filename += '.csv'
    folder = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        raise FileNotFoundError(f"File {filename} not found in {folder}")

if __name__ == "__main__":
    print(f"Loading data from {os.path.dirname(os.path.abspath(__file__))}...")
    data = load_all_data()
    for name, df in data.items():
        print(f"Loaded {name}: {len(df)} samples, {df.columns.tolist()}")
"""

def generate_csv_data():
    print(f"Scanning {INPUT_ROOT}...")
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input directory {INPUT_ROOT} does not exist.")
        return

    processed_count = 0
    os.makedirs(OUTPUT_512, exist_ok=True)
    os.makedirs(OUTPUT_1000, exist_ok=True)

    for patient_folder in os.listdir(INPUT_ROOT):
        patient_path = os.path.join(INPUT_ROOT, patient_folder)
        
        if not os.path.isdir(patient_path):
            continue
            
        print(f"\\nProcessing Patient: {patient_folder}")
        
        # Create output directories for patient
        out_512_patient = os.path.join(OUTPUT_512, patient_folder)
        out_1000_patient = os.path.join(OUTPUT_1000, patient_folder)
        
        os.makedirs(out_512_patient, exist_ok=True)
        os.makedirs(out_1000_patient, exist_ok=True)
        
        # Write loader script
        with open(os.path.join(out_512_patient, 'data_loader.py'), 'w') as f:
            f.write(LOADER_SCRIPT_CONTENT)
        with open(os.path.join(out_1000_patient, 'data_loader.py'), 'w') as f:
            f.write(LOADER_SCRIPT_CONTENT)
        
        files = [f for f in os.listdir(patient_path) if f.upper().endswith(".BIN")]
        
        for file in files:
            src = os.path.join(patient_path, file)
            base_name = os.path.splitext(file)[0]
            
            try:
                ch1, ch2, ch3, ch4, num_samples = extract_data(src)
                
                # Create 512Hz Data
                time_512 = np.arange(num_samples) / 512.0
                df_512 = pd.DataFrame({
                    "Time_s": time_512,
                    "Ch1": ch1,
                    "Ch2": ch2,
                    "Ch3": ch3,
                    "Ch4": ch4
                })
                df_512.to_csv(os.path.join(out_512_patient, f"{base_name}.csv"), index=False)
                
                # Create 1000Hz Data using linear interpolation
                # Calculate new number of samples to match duration exactly
                duration = time_512[-1] if len(time_512) > 0 else 0
                time_1000 = np.arange(0, duration, 1/1000.0)
                
                if len(time_1000) > 0:
                    ch1_1000 = np.interp(time_1000, time_512, ch1)
                    ch2_1000 = np.interp(time_1000, time_512, ch2)
                    ch3_1000 = np.interp(time_1000, time_512, ch3)
                    ch4_1000 = np.interp(time_1000, time_512, ch4)
                    
                    df_1000 = pd.DataFrame({
                        "Time_s": time_1000,
                        "Ch1": ch1_1000,
                        "Ch2": ch2_1000,
                        "Ch3": ch3_1000,
                        "Ch4": ch4_1000
                    })
                    df_1000.to_csv(os.path.join(out_1000_patient, f"{base_name}.csv"), index=False)

                print(f"  [OK] Processed {file} -> 512Hz and 1000Hz")
                processed_count += 1
            except Exception as e:
                print(f"  [ERR] Failed to process {file}: {e}")

    print(f"\\nBatch processing complete. Total files processed: {processed_count}")

if __name__ == "__main__":
    generate_csv_data()
