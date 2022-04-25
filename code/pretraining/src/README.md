# Pretraining Language Model

The selected pretraining dataset is a concatenation of
two corpora: Our text mining corpus and selected corpus from parts of OSCAR dataset. Details about two corpora you can read [paper]()

| Dataset         |       #sent    |   Domain     |
| :---          |       :----:      |    :----:     |
| Our  |       4.7M       |   Health, Medical       |
| OSCAR's selected |       25M       |   Health, Medical, General       |

Several preprocessed sentences in our text mining 
```
Đau mắt đỏ , nhức_nhối khó_chịu Lá rau_ngót 50g , rễ cỏ_xước 30g , lá dâu 30g , lá tre 30g , rau_má 30g , lá chanh 10g .

Tất_cả đều dùng tươi , sắc đặc , chắt lấy nước uống nhiều lần trong ngày .

Vắt lấy chừng 100ml nước , chia hai lần uống mỗi lần cách nhau 10 phút .

Màu_sắc của vỏ trứng không_chỉ ra các giá_trị dinh_dưỡng hoặc chất_lượng của trứng , mà do giống gà đẻ ra nó .

Chỉ có 1 trong mỗi 20.000 quả trứng có_thể chứa các vi_khuẩn nhiễm_độc Salmonella .

Ăn thật nhiều rau xanh , trái_cây : Các vitamin E , C , A cùng với các chất chống ôxy hoá khác có trong rau xanh , trái_cây như cà_rốt , cà_chua , cam , bưởi ... sẽ chống lại các gốc tự_do , ngăn_chặn quá_trình già đi của tế_bào , trẻ_hoá da từ bên trong .

Bởi trong nước bưởi có chứa chất Pyranocoumarin làm tăng_cường chuyển_hoá cytochromes P450 ( men ruột ) gây nên những tác_dụng như : Làm tăng độc_tính của thuốc_lá , nicotin và ethanol , gây hại cho sức_khoẻ .

```

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
                  --save_steps 20000 \
                  --logging_steps 20000 \
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
