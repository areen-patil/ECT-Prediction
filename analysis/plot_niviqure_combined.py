
import struct
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_and_plot_combined(file_path):
    print(f"Processing file: {file_path}")
    
    with open(file_path, 'rb') as f:
        content = f.read()

    # 1. Skip Header (Starts at offset 16)
    start_idx = 16
    
    # 2. Find Footer "NQR" 
    footer_idx = content.find(b'NQR', start_idx)
    
    if footer_idx == -1:
        print("Warning: Footer 'NQR' not found. Reading till end of file.")
        data_block = content[start_idx:]
    else:
        print(f"Footer found at index {footer_idx}. Ignoring data after this.")
        data_block = content[start_idx:footer_idx]

    num_samples = len(data_block) // 16
    print(f"Found {num_samples} samples.")

    ch1_AC = []
    ch2_AC = []
    ch3_AC = []
    
    ch1_DC = []
    ch2_DC = []
    ch3_DC = []
    
    ch4 = [] # Motor (Presumably already has DC?)

    for i in range(num_samples):
        offset = i * 16
        # Read all 8 channels (16 bytes)
        vals = struct.unpack('<8H', data_block[offset : offset+16])
        
        # AC Components (High Pass Filtered - Spikes)
        ch1_AC.append(vals[0])
        ch2_AC.append(vals[1])
        ch3_AC.append(vals[2])
        ch4.append(vals[3]) # Motor usually kept raw
        
        # DC Components (Drift / Baseline)
        ch1_DC.append(vals[4]) # Ch5
        ch2_DC.append(vals[5]) # Ch6
        ch3_DC.append(vals[6]) # Ch7
        # Ch8 (vals[7]) ignored for now as Ch4 seemed fine

    # Convert to Numpy Arrays for vector addition
    ch1_AC = np.array(ch1_AC)
    ch2_AC = np.array(ch2_AC)
    ch3_AC = np.array(ch3_AC)
    
    ch1_DC = np.array(ch1_DC)
    ch2_DC = np.array(ch2_DC)
    ch3_DC = np.array(ch3_DC)
    
    # Reconstruct Full Signal: AC + DC
    # Since both are uint16, adding them might mimic the raw ADC value reconstruction
    # Or maybe it's offset relative to center?
    # Simple addition is the most logical "Compression" method (signal = baseline + delta)
    # But let's check ranges. If AC is centered at ~800 and DC is ~800, adding them gives ~1600.
    # If the original was 0-4096, this makes sense.
    
    final_ch1 = ch1_AC + ch1_DC
    final_ch2 = ch2_AC + ch2_DC
    final_ch3 = ch3_AC + ch3_DC
    final_ch4 = np.array(ch4)

    # 4. Plotting
    sampling_rate = 512 # Hz
    time_sec = np.array(range(num_samples)) / sampling_rate

    plt.figure(figsize=(15, 10))
    
    filename = os.path.basename(file_path)
    
    # Subplot for Ch1 - Combined (AC+DC)
    plt.subplot(4, 1, 1)
    # Using element-wise addition of numpy arrays (already done above)
    plt.plot(time_sec, final_ch1, color='blue', linewidth=0.5)
    plt.ylabel('Channel 1')
    plt.title(f'NIVIQURE Data Reconstructed (AC+DC) - {filename} (Fs=512Hz)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Subplot for Ch2 - Combined (AC+DC)
    plt.subplot(4, 1, 2)
    plt.plot(time_sec, final_ch2, color='green', linewidth=0.5)
    plt.ylabel('Channel 2')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Subplot for Ch3 - Combined (AC+DC)
    plt.subplot(4, 1, 3)
    plt.plot(time_sec, final_ch3, color='#008080', linewidth=0.5) # Teal
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
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f'{base_name}_{sampling_rate}Hz.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Use the file we were inspecting
    FILE_PATH = r"d:/forKrishna/ECT_EEG/ECT_EEG/SZPCRC11004/2ASZPCRC11004.BIN"
    parse_and_plot_combined(FILE_PATH)
