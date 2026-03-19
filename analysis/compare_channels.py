
import struct
import numpy as np

def compare_roughness():
    file_path = r"d:/forKrishna/ECT_EEG/ECT_EEG/SZPCRC11004/2ASZPCRC11004.BIN"
    
    with open(file_path, 'rb') as f:
        content = f.read()

    start_idx = 16
    footer_idx = content.find(b'NQR', start_idx)
    data_block = content[start_idx:footer_idx]
    num_samples = len(data_block) // 16
    
    ch_data = [[], [], [], [], [], [], [], []]
    
    for i in range(num_samples):
        offset = i * 16
        vals = struct.unpack('<8H', data_block[offset : offset+16])
        for ch in range(8):
            ch_data[ch].append(vals[ch])

    ch1 = np.array(ch_data[0]) 
    ch5 = np.array(ch_data[4])
    
    # Calculate 'Roughness' (Sum of absolute differences between consecutive points)
    diff1 = np.abs(np.diff(ch1))
    diff5 = np.abs(np.diff(ch5))
    
    print(f"Roughness Score (High Freq Content):")
    print(f"  Ch1 (Expected Spikes): {np.mean(diff1):.4f}")
    print(f"  Ch5 (Placeholder):     {np.mean(diff5):.4f}")
    
    correlation = np.corrcoef(ch1, ch5)[0, 1]
    print(f"\nCorrelation between Ch1 and Ch5: {correlation:.4f}")

if __name__ == "__main__":
    compare_roughness()
