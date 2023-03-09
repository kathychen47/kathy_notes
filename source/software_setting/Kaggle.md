[toc]

# Kaggle

## Download Kaggle dataset in Linux server

1. install kaggle in Linux

   ```python
   pip install kaggle
   ```



2. get Kaggle API

   Kaggle -> Account -> Create New API Token -> download kaggle.json -> move kaggle. json file to home/user/.kaggle in linux



3. Change download directory

   ```python
   kaggle config set -n path -v /data/XRay_data
   ```

   

4. download dataset

   ```python
   kaggle competitions download -c rsna-2022-cervical-spine-fracture-detection
   ```

   



















