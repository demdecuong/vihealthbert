# Pretraining Language Model


## Data
1) Download
    - You can download raw data and preprocess by yourself
    - Raw data : in [drive path](https://drive.google.com/drive/folders/1oMxBXRqgr9hpSfc__bJ3EftgCNM5NXqG)
2) Processed data:   
    - Has been cleaned and formated in sentences for training MLM
    - Download [here](https://drive.google.com/file/d/13EGlnJvQBq1idXTQaqaZWlN1A5Q9gv8V/view?usp=sharing)  


## Preprocess

- Make data is available for training task
- TBD

## Usage  
```
bash main.sh
```

### Train Masked Token
```
export lr=1e-5
export s=42
echo "${lr}"
export MODEL_DIR=ViHnBERT
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
python3 main.py --token_level word-level \
                  --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir ../data \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 1 \
                  --logging_steps 1 \
                  --num_train_epochs 5 \
                  --train_batch_size 2 \
                  --eval_batch_size 2 \
                  --gpu_id 0 \
                  --learning_rate $lr \
```
### Train Multi task  
- Train NSP and Capitalized Prediction  
```
export lr=1e-5
export s=42
echo "${lr}"
export MODEL_DIR=HnBert
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
python3 main.py --token_level word-level \
                  --model_type hnbert \
                  --model_dir $MODEL_DIR \
                  --data_dir ./data \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 20000 \
                  --logging_steps 20000 \
                  --num_train_epochs 5 \
                  --train_batch_size 2 \
                  --eval_batch_size 2 \
                  --gpu_id 0 \
                  --learning_rate $lr \
                  --do_cap \
                  --cap_coef 0.15 \
                  --do_nsp \
                  --nsp_coef 0.15 
```

## Data:
##### Folder structure:

```
├── data/
|   ├── train/corpus.txt
|   ├── test/corpus.txt
├── dataset.py
├── main.py #training or evaluate model
├── trainer.py # Trainer
├── utis.py
└── ...
```

## Citation
- This work has been done by Tran Hoang Vu and Nguyen Phuc Minh during working at Vinbrain.
