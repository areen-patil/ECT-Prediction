
import struct
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all_8_channels(file_path):
    print(f"Processing all 8 channels for: {file_path}")
    
    with open(file_path, 'rb') as f:
        content = f.read()

    start_idx = 16
    footer_idx = content.find(b'NQR', start_idx)
    data_block = content[start_idx:footer_idx]
    num_samples = len(data_block) // 16
    
    ch_data = [[], [], [], [], [], [], [], []]
    
    for i in range(num_samples):
        offset = i * 16
        # Read all 16 bytes -> 8 shorts
        vals = struct.unpack('<8H', data_block[offset : offset+16])
        for ch in range(8):
            ch_data[ch].append(vals[ch])

    # Plot
    sampling_rate = 512
    time_sec = np.array(range(num_samples)) / sampling_rate
    
    plt.figure(figsize=(15, 20))
    
    colors = ['blue', 'green', 'teal', 'brown', 'purple', 'orange', 'cyan', 'magenta']
    
    for i in range(8):
        plt.subplot(8, 1, i+1)
        plt.plot(time_sec, ch_data[i], color=colors[i], linewidth=0.5)
        plt.ylabel(f'Ch {i+1}')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Consistent Y-limit to compare amplitude
        # plt.ylim(0, 1024) 
        
        if i == 7:
            plt.xlabel('Time (s)')
            
    plt.tight_layout()
    plt.savefig('all_8_channels.png')
    print("Saved all_8_channels.png")

if __name__ == "__main__":
    plot_all_8_channels(r"d:/forKrishna/ECT_EEG/ECT_EEG/SZPCRC11004/2ASZPCRC11004.BIN")
