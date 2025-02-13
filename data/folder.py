import os
import shutil

def move_files_and_delete_subfolders(main_folder):
    if not os.path.exists(main_folder):
        print(f"Der Ordner {main_folder} existiert nicht.")
        return
    
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Verschiebe alle Dateien in den Hauptordner
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path):
                    new_path = os.path.join(main_folder, file_name)
                    shutil.move(file_path, new_path)
            
            # Lösche den leeren Unterordner
            shutil.rmtree(subfolder_path)
            print(f"Gelöscht: {subfolder_path}")
    
    print("Alle Dateien wurden verschoben und die Unterordner gelöscht.")

# Beispiel: Hauptordner angeben
main_folder = f"original_data{os.sep}orig"
move_files_and_delete_subfolders(main_folder)
