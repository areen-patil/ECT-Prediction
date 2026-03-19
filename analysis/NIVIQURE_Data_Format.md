
# NIVIQURE EEG Data Format Specification
**Reverse-Engineered Documentation**
*Date: February 19, 2026*

## 1. Overview
The NIVIQURE EXCOR-SL device stores EEG and ECT data in a custom binary format (`.BIN`). The data is stored in **segments** (e.g., `2A...`, `2B...`) corresponding to different phases of the recording or specific events like seizures. The raw signal is split into two components: **High-Frequency (AC)** and **Low-Frequency (DC/Baseline)** to optimize dynamic range and storage.

## 2. File Format Structure
The file consists of a fixed header, a data body containing interleaved samples, and a footer marked by a specific signature.

### 2.1 Header
*   **Size:** 16 bytes
*   **Content:** Metadata (Exact decoding not fully reversed, likely timestamp/patient ID).
*   **Action:** Should be skipped when reading raw data.

### 2.2 Data Block (The Signal)
*   **Start:** Byte 16
*   **End:** At the "NQR" footer signature.
*   **Sample Size:** 16 bytes per timepoint (sample).
*   **Sampling Frequency:** **512 Hz**
*   **Endianness:** **Little Endian** (`<`)
*   **Data Type:** Unsigned Short (`uint16`, 2 bytes per channel).

#### Channel Layout (16 Bytes per Sample)
Each 16-byte block contains data for 8 channels (4 "AC" + 4 "DC").

| Byte Offset | Format | Component | Description |
| :--- | :--- | :--- | :--- |
| **00 - 01** | `uint16` | **Channel 1 (AC)** | High-pass filtered EEG (Spikes/Waves) |
| **02 - 03** | `uint16` | **Channel 2 (AC)** | High-pass filtered EEG |
| **04 - 05** | `uint16` | **Channel 3 (AC)** | High-pass filtered EEG |
| **06 - 07** | `uint16` | **Channel 4 (Motor)** | EMG / Motor seizure sensor |
| **08 - 09** | `uint16` | **Channel 1 (DC)** | Baseline Drift / DC Offset for Ch 1 |
| **10 - 11** | `uint16` | **Channel 2 (DC)** | Baseline Drift / DC Offset for Ch 2 |
| **12 - 13** | `uint16` | **Channel 3 (DC)** | Baseline Drift / DC Offset for Ch 3 |
| **14 - 15** | `uint16` | **Channel 4 (DC)** | Baseline Drift / DC Offset for Ch 4 (often unused) |

### 2.3 Footer
*   **Signature:** `NQR` (Hex: `4E 51 52`)
*   **Content:** Metadata following the signature (Patient info, settings, etc.).
*   **Action:** Stop reading data when `NQR` is encountered.

---

## 3. Signal Reconstruction
Values in the file are unsigned integers. To reconstruct the visual waveform seen in the software:

1.  **Combine AC and DC:**
    The "True" visual signal is the sum of the AC component and the DC component.
    ```python
    Visual_Signal_Ch1 = Channel_1_AC + Channel_1_DC
    Visual_Signal_Ch2 = Channel_2_AC + Channel_2_DC
    Visual_Signal_Ch3 = Channel_3_AC + Channel_3_DC
    ```

2.  **Y-Axis Inversion:**
    Standard EEG conventions often plot negative potentials upwards.
    *   **Invert the Y-axis** for all channels to match the software's visual orientation.
    *   Alternatively, calculate `Value = MAX_UINT16 - (AC + DC)` directly.

---

## 4. Python Implementation
Below is a standard function to read and plot a NIVIQURE `.BIN` file.

```python
import struct
import numpy as np
import matplotlib.pyplot as plt

def read_niviqure_file(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()

    # 1. Skip Header
    start_idx = 16
    
    # 2. Find Footer
    footer_idx = content.find(b'NQR', start_idx)
    data_block = content[start_idx:footer_idx]
    
    num_samples = len(data_block) // 16
    
    # buffers
    ch1, ch2, ch3, ch4 = [], [], [], []
    
    for i in range(num_samples):
        offset = i * 16
        # Unpack 8 unsigned shorts (Little Endian)
        vals = struct.unpack('<8H', data_block[offset : offset+16])
        
        # reconstruct Signal (AC + DC)
        ch1.append(vals[0] + vals[4]) # Ch 1
        ch2.append(vals[1] + vals[5]) # Ch 2
        ch3.append(vals[2] + vals[6]) # Ch 3
        ch4.append(vals[3])           # Ch 4 (Motor) - usually raw
        
    return np.array(ch1), np.array(ch2), np.array(ch3), np.array(ch4)

def plot_data(ch1, ch2, ch3, ch4, fs=512):
    time = np.arange(len(ch1)) / fs
    plt.figure(figsize=(15, 10))
    
    for i, data in enumerate([ch1, ch2, ch3, ch4]):
        plt.subplot(4, 1, i+1)
        plt.plot(time, data, linewidth=0.5)
        plt.ylabel(f'Ch {i+1}')
        plt.gca().invert_yaxis() # Important for visual match
        plt.grid(True)
        
    plt.show()
```

## 5. Notes
*   **Sample Rate:** 512 Hz is exact.
*   **Duration:** File `2ASZPCRC11004.BIN` corresponds to ~265 seconds (4.4 mins).
*   **Seizure Duration:** Can be estimated by analyzing the variance (activity) of the AC components (Bytes 0-5).
