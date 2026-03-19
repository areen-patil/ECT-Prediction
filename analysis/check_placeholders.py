
import struct
import matplotlib.pyplot as plt
import numpy as np
import os

def check_placeholders(file_path):
    print(f"Checking placeholders in: {file_path}")
    
    with open(file_path, 'rb') as f:
        content = f.read()

    start_idx = 16
    footer_idx = content.find(b'NQR', start_idx)
    data_block = content[start_idx:footer_idx]
    num_samples = len(data_block) // 16
    
    ch5 = [] # Bytes 08-09
    ch6 = [] # Bytes 10-11
    ch7 = [] # Bytes 12-13
    ch8 = [] # Bytes 14-15 (Expected 00)
    
    for i in range(num_samples):
        offset = i * 16
        # Extract the SECOND 8 bytes (previously ignored)
        placeholder_bytes = data_block[offset+8 : offset+16]
        
        vals = struct.unpack('<4H', placeholder_bytes)
        ch5.append(vals[0])
        ch6.append(vals[1])
        ch7.append(vals[2])
        ch8.append(vals[3])

    # Plot
    sampling_rate = 512
    time_sec = np.array(range(num_samples)) / sampling_rate
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(4,1,1)
    plt.plot(time_sec, ch5, label='Bytes 08-09')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4,1,2)
    plt.plot(time_sec, ch6, label='Bytes 10-11')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4,1,3)
    plt.plot(time_sec, ch7, label='Bytes 12-13')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4,1,4)
    plt.plot(time_sec, ch8, label='Bytes 14-15')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('placeholder_check.png')
    print("Saved placeholder_check.png")

if __name__ == "__main__":
    check_placeholders(r"d:/forKrishna/ECT_EEG/ECT_EEG/SZPCRC11004/2ASZPCRC11004.BIN")
