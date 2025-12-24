import os
import shutil
import time
import random

HOME_DIR = os.path.expanduser("~")
dst_dir = "second_data"  # 실행 전에 확인!!!

def split_files(source_directory, train_dir=os.path.join(HOME_DIR, dst_dir, "train/images"),
                validation_dir=os.path.join(HOME_DIR, dst_dir, "valid/images"),
                test_dir=os.path.join(HOME_DIR, dst_dir, "test/images"),
                train_ratio=0.7, validation_ratio=0.2):
    
    # Ensure ratios sum to 1
    if (train_ratio + validation_ratio) > 1:
        print("Error: Ratios sum to more than 1.")
        return

    # Create destination directories if they do not exist
    for dir_name in [train_dir, validation_dir, test_dir]:
        # os.makedirs(os.path.join(source_directory, dir_name), exist_ok=True)
        os.makedirs(os.path.join(dir_name), exist_ok=True)

    # Get all files in the source directory
    all_files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]
    
    random.seed(time.time())
    random.shuffle(all_files)  # Shuffle to randomize file selection

    # Calculate split indices
    total_files = len(all_files)
    train_end = int(total_files * train_ratio)
    validation_end = train_end + int(total_files * validation_ratio)

    # Split files
    train_files = all_files[:train_end]
    validation_files = all_files[train_end:validation_end]
    test_files = all_files[validation_end:]

    # Function to copy files to destination directories
    def copy_files(files, destination):
        for file in files:
            shutil.copy(os.path.join(source_directory, file), 
                        os.path.join(destination, file))

    # Copy files to their respective directories
    copy_files(train_files, train_dir)
    copy_files(validation_files, validation_dir)
    copy_files(test_files, test_dir)

    print(f"Files split into {train_dir}, {validation_dir}, and {test_dir} directories.")

def main():
    split_files(os.path.join(HOME_DIR, "data"))

if __name__ == "__main__":
    main()