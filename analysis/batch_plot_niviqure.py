
import struct
import matplotlib.pyplot as plt
import numpy as np
import os

INPUT_ROOT = r"d:/forKrishna/ECT_EEG/ECT_EEG"
OUTPUT_ROOT = r"d:/forKrishna/ECT_EEG/plots"

def parse_and_plot(file_path, output_path):
    # print(f"Processing file: {file_path}")
    
    with open(file_path, 'rb') as f:
        content = f.read()

    # 1. Skip Header (Starts at offset 16)
    start_idx = 16
    
    # 2. Find Footer "NQR" 
    footer_idx = content.find(b'NQR', start_idx)
    
    # Simple check if footer exists
    if footer_idx != -1:
        data_block = content[start_idx:footer_idx]
    else:
        # print("Warning: Footer 'NQR' not found. Reading till end of file.")
        data_block = content[start_idx:]

    num_samples = len(data_block) // 16
    
    # Buffers
    ch1_AC, ch2_AC, ch3_AC = [], [], []
    ch1_DC, ch2_DC, ch3_DC = [], [], []
    ch4 = [] 

    for i in range(num_samples):
        offset = i * 16
        vals = struct.unpack('<8H', data_block[offset : offset+16])
        
        # AC Components (High Pass Filtered - Spikes)
        ch1_AC.append(vals[0])
        ch2_AC.append(vals[1])
        ch3_AC.append(vals[2])
        ch4.append(vals[3]) 
        
        # DC Components (Drift / Baseline)
        ch1_DC.append(vals[4]) 
        ch2_DC.append(vals[5]) 
        ch3_DC.append(vals[6]) 

    # Vector Math
    final_ch1 = np.array(ch1_AC) + np.array(ch1_DC)
    final_ch2 = np.array(ch2_AC) + np.array(ch2_DC)
    final_ch3 = np.array(ch3_AC) + np.array(ch3_DC)
    final_ch4 = np.array(ch4)

    # Plotting
    sampling_rate = 512 # Hz
    time_sec = np.array(range(num_samples)) / sampling_rate

    plt.figure(figsize=(15, 10))
    
    filename = os.path.basename(file_path)
    
    # Subplot for Ch1
    plt.subplot(4, 1, 1)
    plt.plot(time_sec, final_ch1, color='blue', linewidth=0.5)
    plt.ylabel('Channel 1')
    plt.title(f'NIVIQURE Data Reconstructed (AC+DC) - {filename} (Fs=512Hz)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Subplot for Ch2
    plt.subplot(4, 1, 2)
    plt.plot(time_sec, final_ch2, color='green', linewidth=0.5)
    plt.ylabel('Channel 2')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Subplot for Ch3
    plt.subplot(4, 1, 3)
    plt.plot(time_sec, final_ch3, color='#008080', linewidth=0.5) 
    plt.ylabel('Channel 3')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Subplot for Ch4
    plt.subplot(4, 1, 4)
    plt.plot(time_sec, final_ch4, color='brown', linewidth=0.5)
    plt.ylabel('Channel 4')
    plt.xlabel('Time (seconds)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def process_all():
    print(f"Scanning {INPUT_ROOT}...")
    
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input directory {INPUT_ROOT} does not exist.")
        return

    processed_count = 0
    
    # Walk through the directory structure
    for patient_folder in os.listdir(INPUT_ROOT):
        patient_path = os.path.join(INPUT_ROOT, patient_folder)
        
        # Skip files in root, only look at directories
        if not os.path.isdir(patient_path):
            continue
            
        print(f"\nFound Patient Folder: {patient_folder}")
        
        # Create corresponding output folder
        output_patient_dir = os.path.join(OUTPUT_ROOT, patient_folder)
        os.makedirs(output_patient_dir, exist_ok=True)
        
        files = [f for f in os.listdir(patient_path) if f.upper().endswith(".BIN")]
        
        for file in files:
            src = os.path.join(patient_path, file)
            
            # Construct output filename
            base_name = os.path.splitext(file)[0]
            dst_name = f"{base_name}_512Hz.png"
            dst = os.path.join(output_patient_dir, dst_name)
            
            # Skip if already exists? Maybe not, user asked to plot. To save time we could check.
            # But let's overwrite to ensure latest plotting logic.
            
            try:
                parse_and_plot(src, dst)
                print(f"  [OK] Plotted {file} -> {dst_name}")
                processed_count += 1
            except Exception as e:
                print(f"  [ERR] Failed {file}: {e}")

    print(f"\nBatch processing complete. Total files processed: {processed_count}")

if __name__ == "__main__":
    process_all()
