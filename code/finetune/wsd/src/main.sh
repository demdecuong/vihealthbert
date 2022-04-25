export lr=1e-5
export s=3407
echo "${lr}"
export MODEL_DIR=ViHnBERT
export MODEL_DIR=$MODEL_DIR
echo "${MODEL_DIR}"
python3 main.py --token_level word-level \
                  --model_type vihnbert \
                  --model_dir $MODEL_DIR \
                  --data_dir ../data \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 200 \
                  --logging_steps 200 \
                  --num_train_epochs 100 \
                  --tuning_metric macro_f1 \
                  --gpu_id 0 \
                  --learning_rate $lr