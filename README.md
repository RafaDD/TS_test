## Step 1
Download data from [JBOX](https://jbox.sjtu.edu.cn/l/P1b1xw), which is ```transposed_shortened.csv``` and put it at dir ```datasets/raw_data/Paris```

Run command
```
python scripts/data_preparation/Paris/generate_training_data.py
```
to process data.

## Step 2
Train the data with a specific model (DLinear supported).

Run command
```
python experiments/train.py -c baselines/DLinear/Paris.py --gpus '0'
```
and the result will be saved at ```result/```

## Step 3
Run command
```
python spawn_answer.py
```
and the result will be saved at ```result/submit.csv```

Then it is ready for evaluation in Kaggle.