export lr=1e-5
export s=3407
echo "${lr}"
export MODEL_DIR=ViHnBERT
echo "${MODEL_DIR}"
python3 main.py --token_level word-level --model_type phobert --model_dir $MODEL_DIR --data_dir ../data --seed $s --do_train --do_eval --save_steps 100 --logging_steps 100 --train_batch_size 32 --eval_batch_size 40 --gpu_id 0 --learning_rate $lr