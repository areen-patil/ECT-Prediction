
import struct
import numpy as np

def analyze_placeholders():
    file_path = r"d:/forKrishna/ECT_EEG/ECT_EEG/SZPCRC11004/2ASZPCRC11004.BIN"
    
    with open(file_path, 'rb') as f:
        content = f.read()

    start_idx = 16
    footer_idx = content.find(b'NQR', start_idx)
    data_block = content[start_idx:footer_idx]
    num_samples = len(data_block) // 16
    
    ch_data = [[], [], [], []]
    
    for i in range(num_samples):
        offset = i * 16
        # Extract placeholders (bytes 8-15)
        placeholder_bytes = data_block[offset+8 : offset+16]
        vals = struct.unpack('<4H', placeholder_bytes)
        for ch in range(4):
            ch_data[ch].append(vals[ch])
            
    # Calculate stats
    names = ["Bytes 08-09", "Bytes 10-11", "Bytes 12-13", "Bytes 14-15"]
    for i in range(4):
        data = np.array(ch_data[i])
        print(f"\n{names[i]}:")
        print(f"  Min: {np.min(data)}, Max: {np.max(data)}")
        print(f"  Mean: {np.mean(data):.2f}")
        unique_vals = np.unique(data)
        if len(unique_vals) < 10:
            print(f"  Unique Values: {unique_vals}")
        else:
            print(f"  Unique Values Count: {len(unique_vals)}")

if __name__ == "__main__":
    analyze_placeholders()
