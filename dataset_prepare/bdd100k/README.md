# Dataset prepareration
## BDD100K dataset download and create image from movie
`${BDD100k-Path}` is place of dataset which you determined.  

### requiment

- `parallel`
- `aria2c` or `wget`
- `ffmpeg`

### create dataset using multiprocessing
if you want to create quickely, run job by multiprocessing  
<!-- (required command `ybatch` or `qsub` for job command)  -->
<!-- https://github.com/rioyokotalab/video-representation-learning/tree/main/scripts -->

1. download dataset

    ```shell
    bash get_data/download_videos.sh ${BDD100k-Path}
    ```

2. unzip dataset

    ```shell
    bash get_data/unzip_videos.sh ${BDD100k-Path}
    ```

3. create directory for images which you use in training

    ```shell
    bash get_data/mkdir_train_val_img.sh ${BDD100k-Path}
    ```

4. finally, create images using multiprocessing

    Ex. create 1900/process using 37 process(machine)

    1st node  
    ```shell
    bash get_data/create_img.sh ${BDD100k-Path} 1 1900
    ```
    2nd node  
    ```shell
    bash get_data/create_img.sh ${BDD100k-Path} 1901 1900
    ```
    :  
    n-th node
    ```shell
    bash get_data/create_img.sh ${BDD100k-Path} (n-1)*1900+1 1900
    ```
    :  
    37th node  
    ```shell
    bash get_data/create_img.sh ${BDD100k-Path} 68401 1900
    ```


  <!-- 1. create job scripts for multiprocessing

      ```shell
      bash job_sh/gen_job_sh/gen_create_trainval_img_job.sh ${BDD100k-Path} 
      ```

  2. run job by using job scripts which you created above command

      ```shell
      bash job_sh/sub_job_sh/train_val_gen_img_job_sub.sh
      ``` -->

### create dataset using singleprocess
if you don't mind time to create dataset, run following command

```shell
bash process_bdd.sh ${BDD100k-Path}
```

## Final data structure
data structure after completing the above instructions

```python
${BDD100k-Path}
 |-- bdd100k
 |   |-- videos # 1.5TB
 |   |   |-- train # 1.3TB
 |   |   |-- val # 184GB
 |   |-- images # 3.5TB
 |   |   |-- train # 3.1TB
 |   |   |-- val # 443GB
```
