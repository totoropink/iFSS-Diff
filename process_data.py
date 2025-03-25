import os
from PIL import Image
import numpy as np

def convert_images(directory):
    print(f"Processing directory: {directory}")
    
    for root, dirs, files in os.walk(directory):
        print(f"Checking files in {root}")
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"Found file: {filename}")

            if filename.endswith('.png'):  
                image = Image.open(file_path).convert('L')
                image_array = np.array(image)

                print(f"Min pixel value: {np.min(image_array)}")
                print(f"Max pixel value: {np.max(image_array)}")
                print(f"Unique values in image: {np.unique(image_array)}")

                image_array = np.where(image_array >= 128, 1, 0)
            
                print(f"{filename} - unique values in array: {np.unique(image_array)}")

                new_filename = filename[:-4] + '.npy'
                output_file_path = os.path.join(root, new_filename)
                np.save(output_file_path, image_array)
                print(f"Converted {filename} to {output_file_path}")

            if filename.endswith('.jpg'):  
                image = Image.open(file_path).convert('RGB')  
                new_filename = filename[:-4] + '.png'
                output_file_path = os.path.join(root, new_filename)
                image.save(output_file_path, "PNG")
                print(f"Converted {filename} to {output_file_path}")
                os.remove(file_path)
                print(f"Deleted original file {filename}")

def process_data(base_directory):
    print(f"Starting to process base directory: {base_directory}")
    for folder in range(4):  
        folder_path = os.path.join(base_directory, str(folder))
        if os.path.exists(folder_path):
            print(f"Processing folder {folder_path}")
            for category_name in os.listdir(folder_path):  
                category_path = os.path.join(folder_path, category_name, 'train')
                if os.path.exists(category_path):
                    print(f"Found category directory: {category_path}")
                    convert_images(category_path)
                else:
                    print(f"Category directory not found: {category_path}")
        else:
            print(f"Folder path not found: {folder_path}")

# 主程序入口
base_directory = 'pascal-5/'
process_data(base_directory)
