import os
import shutil

# This is the directory that SHOULD contain the patient folders
BASE_DIR = r"d:/forKrishna/ECT_EEG/ECT_EEG"

def fix_nested_structure():
    """
    Checks if there is an extra 'ECT_EEG' folder inside BASE_DIR.
    If so, moves all contents from BASE_DIR/ECT_EEG to BASE_DIR
    and removes the empty 'ECT_EEG' folder.
    """
    nested_dir = os.path.join(BASE_DIR, "ECT_EEG")
    
    if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
        print(f"Detected nested directory: {nested_dir}")
        print("Moving contents to base directory...")
        
        items = os.listdir(nested_dir)
        for item in items:
            src = os.path.join(nested_dir, item)
            dst = os.path.join(BASE_DIR, item)
            
            try:
                shutil.move(src, dst)
                # print(f"Moved {item}")
            except Exception as e:
                print(f"Error moving {item}: {e}")
        
        try:
            os.rmdir(nested_dir)
            print("Removed empty nested directory.")
        except Exception as e:
            print(f"Error removing nested directory: {e}")
    else:
        print("No nested 'ECT_EEG' directory found (or already fixed).")

def cleanup_patients():
    """
    Iterates through patient folders in BASE_DIR.
    - Recursively finds files starting with '2'.
    - Moves them to the patient root.
    - Deletes patient folder if no '2...' files found.
    - Cleans up non-'2...' files and subdirectories.
    """
    print(f"Scanning for patient folders in: {BASE_DIR}")
    
    if not os.path.exists(BASE_DIR):
        print(f"Base directory does not exist: {BASE_DIR}")
        return

    patient_items = os.listdir(BASE_DIR)
    
    deleted_patients = 0
    kept_patients = 0
    
    for item in patient_items:
        patient_path = os.path.join(BASE_DIR, item)
        
        # We only care about directories (patient folders)
        if not os.path.isdir(patient_path):
            continue
            
        # Recursive search for '2...' files
        valid_files = []
        for root, dirs, files in os.walk(patient_path):
            for file in files:
                if file.startswith('2'):
                    valid_files.append(os.path.join(root, file))
        
        # If no valid files, delete the patient folder
        if not valid_files:
            print(f"Deleting patient {item} (No files starting with '2')")
            try:
                shutil.rmtree(patient_path)
                deleted_patients += 1
            except Exception as e:
                print(f"Error deleting {item}: {e}")
            continue
            
        print(f"Processing patient {item}: Found {len(valid_files)} valid files.")
        
        # Move valid files to patient root
        for src_path in valid_files:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(patient_path, filename)
            
            if src_path != dst_path:
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Error moving file {filename}: {e}")
        
        # Cleanup: Delete everything in patient folder that is NOT a '2...' file
        # Re-list items in the patient root
        current_items = os.listdir(patient_path)
        for sub_item in current_items:
            sub_path = os.path.join(patient_path, sub_item)
            
            # If it's a directory, it's junk (since we moved files out)
            if os.path.isdir(sub_path):
                try:
                    shutil.rmtree(sub_path)
                except Exception as e:
                    print(f"Error removing directory {sub_item}: {e}")
            else:
                # If it's a file, check if it starts with '2'
                if not sub_item.startswith('2'):
                    try:
                        os.remove(sub_path)
                    except Exception as e:
                        print(f"Error removing file {sub_item}: {e}")
        
        kept_patients += 1
        
    print("-" * 30)
    print(f"Cleanup Complete.")
    print(f"Patients kept: {kept_patients}")
    print(f"Patients deleted: {deleted_patients}")

if __name__ == "__main__":
    fix_nested_structure()
    cleanup_patients()
