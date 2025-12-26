import os
import shutil

def move_corresponding_txt_files(source_dir_with_jpg, source_dir_with_txt, target_dir):
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Get list of .jpg filenames in the source directory
    jpg_files = [f for f in os.listdir(source_dir_with_jpg) if f.endswith('.jpg')]

    # Iterate over each .jpg file to find corresponding .txt file
    for jpg_file in jpg_files:
        # Construct .txt filename based on the .jpg filename
        base_filename = os.path.splitext(jpg_file)[0]
        txt_filename = base_filename + '.txt'

        # Construct full path for the .txt file in the source directory
        txt_file_path = os.path.join(source_dir_with_txt, txt_filename)
        # txt_file_path = os.path.join(source_dir_with_txt, 
        #                              txt_filename).replace('\\', '/')
        # print(txt_file_path)


        # Check if the .txt file exists
        if os.path.exists(txt_file_path):
            # Move the .txt file to the target directory
            shutil.copy(txt_file_path, os.path.join(target_dir, txt_filename))
            print(f"Copied: {txt_filename}")
        else:
            print(f"Not found: {txt_filename}")


def main():
    HOME_DIR = os.path.expanduser("~")
    dst_dir = "third_data"  # 실행 전에 확인!!!
    source_dir_with_jpg = os.path.join(HOME_DIR, f'{dst_dir}/train/images')
    source_dir_with_txt = os.path.join(HOME_DIR, 'labels')
    target_dir = os.path.join(HOME_DIR, f'{dst_dir}/train/labels')

    move_corresponding_txt_files(source_dir_with_jpg, source_dir_with_txt, target_dir)

    source_dir_with_jpg = os.path.join(HOME_DIR, f'{dst_dir}/valid/images')
    target_dir = os.path.join(HOME_DIR, f'{dst_dir}/valid/labels')

    move_corresponding_txt_files(source_dir_with_jpg, source_dir_with_txt, target_dir)

if __name__ == "__main__":
    main()