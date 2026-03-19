import os
import pandas as pd
import matplotlib.pyplot as plt

# Define directories
INPUT_512 = r"d:/forKrishna/ECT_EEG/csv_512Hz"
INPUT_1000 = r"d:/forKrishna/ECT_EEG/csv_1000Hz"
OUTPUT_PLOTS_512 = r"d:/forKrishna/ECT_EEG/plots_csv_512Hz"
OUTPUT_PLOTS_1000 = r"d:/forKrishna/ECT_EEG/plots_csv_1000Hz"

def plot_csv_data(csv_path, output_path, title_prefix, is_1000hz=False):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract data columns
    time_sec = df['Time_s'].values
    ch1 = df['Ch1'].values
    ch2 = df['Ch2'].values
    ch3 = df['Ch3'].values
    ch4 = df['Ch4'].values

    # Determine frequency string for titles/labels
    freq_str = "1000Hz" if is_1000hz else "512Hz"

    # Setup the plot
    plt.figure(figsize=(15, 10))
    
    filename = os.path.basename(csv_path)
    
    # Subplot for Ch1
    plt.subplot(4, 1, 1)
    plt.plot(time_sec, ch1, color='blue', linewidth=0.5)
    plt.ylabel('Channel 1')
    plt.title(f'{title_prefix} - {filename} (Fs={freq_str})')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()  # Match the original visual orientation

    # Subplot for Ch2
    plt.subplot(4, 1, 2)
    plt.plot(time_sec, ch2, color='green', linewidth=0.5)
    plt.ylabel('Channel 2')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Subplot for Ch3
    plt.subplot(4, 1, 3)
    plt.plot(time_sec, ch3, color='#008080', linewidth=0.5) # Teal
    plt.ylabel('Channel 3')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Subplot for Ch4
    plt.subplot(4, 1, 4)
    plt.plot(time_sec, ch4, color='brown', linewidth=0.5)
    plt.ylabel('Channel 4')
    plt.xlabel('Time (seconds)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()

    # Save and cleanup
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def process_directory(input_root, output_root, title_prefix, is_1000hz=False):
    print(f"Scanning {input_root}...")
    if not os.path.exists(input_root):
        print(f"Error: Input directory {input_root} does not exist.")
        return

    processed_count = 0
    os.makedirs(output_root, exist_ok=True)

    for patient_folder in os.listdir(input_root):
        patient_path = os.path.join(input_root, patient_folder)
        
        if not os.path.isdir(patient_path):
            continue
            
        print(f"Processing Patient: {patient_folder}")
        
        # Create output directory for patient
        out_patient_dir = os.path.join(output_root, patient_folder)
        os.makedirs(out_patient_dir, exist_ok=True)
        
        files = [f for f in os.listdir(patient_path) if f.upper().endswith(".CSV")]
        
        for file in files:
            src = os.path.join(patient_path, file)
            base_name = os.path.splitext(file)[0]
            dst_name = f"{base_name}_{'1000Hz' if is_1000hz else '512Hz'}.png"
            dst = os.path.join(out_patient_dir, dst_name)
            
            try:
                plot_csv_data(src, dst, title_prefix, is_1000hz)
                print(f"  [OK] Plotted {file} -> {dst_name}")
                processed_count += 1
            except Exception as e:
                print(f"  [ERR] Failed to plot {file}: {e}")

    print(f"Directory processing complete. Total files plotted: {processed_count}")


def generate_all_plots():
    print("--- Generating 512Hz Plots ---")
    process_directory(INPUT_512, OUTPUT_PLOTS_512, "CSV Reconstructed Data", is_1000hz=False)
    
    print("\\n--- Generating 1000Hz Plots ---")
    process_directory(INPUT_1000, OUTPUT_PLOTS_1000, "CSV Interpolated Data", is_1000hz=True)
    
    print("\\nAll plotting tasks complete!")

if __name__ == "__main__":
    generate_all_plots()
