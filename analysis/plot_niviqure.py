
import struct
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_and_plot(file_path):
    print(f"Processing file: {file_path}")
    
    with open(file_path, 'rb') as f:
        content = f.read()

    # 1. Skip Header (Starts at offset 16)
    start_idx = 16
    
    # 2. Find Footer "NQR" to determine end index
    # Search for the byte sequence matching "NQR" (0x4E 0x51 0x52)
    footer_idx = content.find(b'NQR', start_idx)
    
    if footer_idx == -1:
        print("Warning: Footer 'NQR' not found. Reading till end of file.")
        data_block = content[start_idx:]
    else:
        print(f"Footer found at index {footer_idx}. Ignoring data after this.")
        data_block = content[start_idx:footer_idx]

    # 3. Parse Data in 16-byte chunks
    # We ignore the last partial chunk if any
    num_samples = len(data_block) // 16
    print(f"Found {num_samples} samples.")

    ch1 = []
    ch2 = []
    ch3 = []
    ch4 = []

    for i in range(num_samples):
        offset = i * 16
        # Extract first 8 bytes of the 16-byte block
        # The remaining 8 bytes (offset+8 to offset+16) are ignored placeholders
        sample_bytes = data_block[offset : offset + 8]
        
        # Unpack 4 unsigned short integers (2 bytes each)
        # '<' = Little Endian
        # 'H' = Unsigned Short (0-65535)
        # '4H' = 4 shorts
        vals = struct.unpack('<4H', sample_bytes)
        
        ch1.append(vals[0])
        ch2.append(vals[1])
        ch3.append(vals[2])
        ch4.append(vals[3])

    # 4. Plotting
    # Create time axis with Fs = 512 Hz
    sampling_rate = 512 # Hz
    time_sec = np.array(range(num_samples)) / sampling_rate

    plt.figure(figsize=(15, 10))
    
    # Subplot for Ch1
    plt.subplot(4, 1, 1)
    plt.plot(time_sec, ch1, color='blue', linewidth=0.8)
    plt.ylabel('Channel 1')
    plt.title(f'NIVIQURE Data Plot - {os.path.basename(file_path)} (Fs=512Hz)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Subplot for Ch2
    plt.subplot(4, 1, 2)
    plt.plot(time_sec, ch2, color='green', linewidth=0.8)
    plt.ylabel('Channel 2')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Subplot for Ch3
    plt.subplot(4, 1, 3)
    plt.plot(time_sec, ch3, color='#008080', linewidth=0.8) # Teal
    plt.ylabel('Channel 3')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Subplot for Ch4
    plt.subplot(4, 1, 4)
    plt.plot(time_sec, ch4, color='brown', linewidth=0.8)
    plt.ylabel('Channel 4')
    plt.xlabel('Time (seconds)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis() # Invert Y-axis for Channel 4

    plt.tight_layout()
    
    output_file = 'niviqure_plot_11008_512Hz.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Use the file we were inspecting
    FILE_PATH = r"d:/forKrishna/ECT_EEG/ECT_EEG/SZPCRC11004/2ASZPCRC11004.BIN"
    parse_and_plot(FILE_PATH)
