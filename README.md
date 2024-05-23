## Step 0
Download the raw data and put it under ```datasets/raw_data/Paris```.

Run command
```bash
python scripts/data_preparation/Paris/pre.py;
python scripts/data_preparation/Paris/merge.py;
```
for pre-processing of the raw data. Note that multiprocessing is used in ```pre.py```, modify the number of parallel processes according to the number of cores of your CPU. The results will be saved in ```datasets/raw_data/Paris/transposed_shortened.csv```. You may download the processed data from [JBOX](https://jbox.sjtu.edu.cn/l/P1b1xw) as well, download ```transposed_shortened.csv``` and put it at dir ```datasets/raw_data/Paris```.

## Step 1
The following steps are following prediction process of [Basic_TS](https://github.com/zezhishao/BasicTS).

Run command
```bash
python scripts/data_preparation/Paris/generate_training_data.py
```
to process data. We simply used average value to replace ```NaN``` values in this process.

## Step 2
Train the data with a specific model, you may explore different model structures by modify the corresponding control file. In this repository, DLinear is supported.

Run command
```bash
python experiments/train.py -c baselines/DLinear/Paris.py --gpus '0'
```
After training, the trained model will produce predictions on test data. The predcitions will be saved at ```result/Paris.npy```

## Step 3
Run command
```bash
python spawn_answer.py
```
to produce submission file that is ready for evaluation. The result will be saved at ```result/submit.csv```

Then it is ready for evaluation in Kaggle.
