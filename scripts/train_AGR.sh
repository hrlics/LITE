python train.py \
    --dataset_folder datasets/Agr \
    --dataset_name Agr \
    --epochs 20 \
    --train_batchsize 16 \
    --test_batchsize 32 \
    --learning_rate 1e-5 \
    --bert_path distilbert/distilbert-base-uncased \
    --vision_path microsoft/swin-tiny-patch4-window7-224 \
    --d_model 768 \
    --SEED 3407 \
    --dropout_rate 0.3 \
    --warm_up_epochs 5 \
    --wandb True \
    --llama_token None

