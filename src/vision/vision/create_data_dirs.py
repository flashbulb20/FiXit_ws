import os


def create_directory(new_dir_path):
    # Create the new directory
    try:
        os.mkdir(new_dir_path)
        print(f"Directory '{new_dir_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{new_dir_path}' already exists.")

# Example usage
HOME_DIR = os.path.expanduser("~")
dir_name = "first_data" # 실행 전에 확인할 것
dst_dir = os.path.join(HOME_DIR, dir_name)
create_directory(dst_dir)


def main():
    train_dir, val_dir, test_dir = os.path.join(dst_dir, "train"), os.path.join(dst_dir, "valid"), os.path.join(dst_dir, "test")
    create_directory(train_dir)
    create_directory(val_dir)
    create_directory(test_dir)
    create_directory(os.path.join(train_dir, "images"))
    create_directory(os.path.join(train_dir, "labels"))
    create_directory(os.path.join(val_dir, "images"))
    create_directory(os.path.join(val_dir, "labels"))
    create_directory(os.path.join(test_dir, "images"))

if __name__ == "__main__":
    main()