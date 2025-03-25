# Reference Documentation



## 1. Environment Setup

Please refer to SLiMe(https://github.com/aliasgharkhani/SLiMe) for environment configuration details.



## 2. Data Processing

1. Download the corresponding dataset: iFSS you can follow the https://github.com/fcdl94/FSS and CD-FSS you can follow the https://github.com/slei109/PATNet?tab=readme-ov-file.

2. organize images and masks following this structure:

   ```
   pascal-5/
   	├──train/
       		├── 0/                                     
       		|   ├── images
       		|   └── masks
       		├── 1/                                   
       		├── ... 
       		├── 3/
   	├──val/
   coco-20/
   	├──train/
       		├── 0/                                     
       		|   ├── images
       		|   └── masks
       		├── 1/                                   
       		├── ... 
       		├── 3/
       ├──val/
   Deepglobe/                                        
       ├── 1/                                     
       |   ├── images
       |   └── masks
       ├── 2/                                   
       ├── ... 
       ├── 6/
   ISIC/                                        
       ├── 1/                                     
       |   ├── images
       |   └── masks
       ├── 2/                                   
       ├── ... 
       ├── 3/
   Lung/
   	├── train/                                     
       |   ├── images
       |   └── masks
   FSS-1000/
   	├── ab_wheel/
       └── ...
   ```

3. Run the script:

   ```
   python process_data.py
   ```



## 3. Training

Execute the training script:

```
python train.py
```



## 4. Testing

Run the testing script: {CHECKPOINT_DIR} is the path where your optimized embeddings are stored, {TEST_DIR} is the test data path, and {OUTPUT_DIR} is the path your want to output results.

```
python -m src.main2 --dataset pascal \ 
					--checkpoint_dir {CHECKPOINT_DIR} \
					--test_data_dir {TEST_DIR} \
					--output_dir {OUTPUT_DIR} \
					--save_test_predictions
```



## Thanks

The implementation is based on https://github.com/aliasgharkhani/SLiMe
