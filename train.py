# -*- coding: utf-8 -*- 
import os 
import sys 
import logging 
from datetime import datetime 
import subprocess

log_dir = "/pascal_log/"  
if not os.path.exists(log_dir): 
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

sys.stdout = open(log_file, "w") 
sys.stderr = open(log_file, "w")

logging.basicConfig(filename=log_file, level=logging.INFO)

logging.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

command_template = (
    "CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com "
    "python -m src.main --dataset_name pascal --part_names background \"{class_name}\" "
    "--train_num_samples 1 --val_num_samples 10"
    "--train_data_dir \"{train_dir}\" --val_data_dir \"{val_dir}\" --test_data_dir \"{test_dir}\" "
    "--train --checkpoint_dir \"{checkpoint_dir}\" --output_dir \"{output_dir}\" --min_crop_ratio 0.6"
)

# 遍历 pascal-5 目录下的 0 到 3
for i in range(1):
    base_dir = f"pascal-5/{i}"  
    output_base_dir = "pascal_output"  

    # 遍历子目录并执行训练
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)

        if os.path.isdir(class_path): 
            train_dir = os.path.join(class_path, "train") 
            val_dir = os.path.join(class_path, "val") 
            test_dir = os.path.join(class_path, "test")

            checkpoint_dir = os.path.join(output_base_dir, f"pascal_{class_name}_1_shot") 
            output_dir = os.path.join(output_base_dir, f"pascal_{class_name}_1_shot")

            if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir): 
                logging.info(f"Skipping {class_name} in {base_dir} as it is already trained.") 
                continue

            command = command_template.format(
                class_name=class_name,
                train_dir=train_dir,
                val_dir=train_dir,
                test_dir=val_dir,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir
            )

            logging.info(f"Executing command for class \"{class_name}\" in {base_dir}: {command}")

            subprocess.run(command, shell=True)

logging.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

sys.stdout.close()
sys.stderr.close()
