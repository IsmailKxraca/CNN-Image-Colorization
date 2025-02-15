import os
import shutil


def move_files_and_delete_subfolders(main_folder):
    if not os.path.exists(main_folder):
        print(f"Der Ordner {main_folder} existiert nicht.")
        return

    # find all data in folder
    for root, _, files in os.walk(main_folder, topdown=False):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(main_folder, file)

            # rename if image with same name
            counter = 1
            while os.path.exists(dst_path):
                name, ext = os.path.splitext(file)
                dst_path = os.path.join(main_folder, f"{name}_{counter}{ext}")
                counter += 1

            shutil.move(src_path, dst_path)

    # delete all subfolders that are empty
    for root, dirs, _ in os.walk(main_folder, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


# example
main_folder = f"/workspace/CNN-Image-Colorization/data/original_data/orig"
move_files_and_delete_subfolders(main_folder)
