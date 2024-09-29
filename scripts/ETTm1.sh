for model_name in ModernTCN Transformer Informer Reformer Flowformer \
    Flashformer iTransformer iInformer iReformer iFlowformer iFlashformer
do
for batch_size in 128 256 512
do
for lr in 0.001
do
for seq_len in 96 192 336
do
for pred_len in 96 192 336 720
do

python -u run.py \
    --is_training 1 \
    --model_id ETTm1 \
    --model $model_name \
    --root_path ./datasets/ETT-small \
    --data_path ETTm1.csv \
    --log_dir ./logs/ \
    --log_name ETTm1.txt \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --ffn_ratio 8 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 3 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.0 \
    --enc_in 7 \
    --dec_in 7 \
    --e_layers 2 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 256 \
    --dropout 0.3 \
    --itr 1 \
    --train_epochs 100 \
    --batch_size $batch_size \
    --patience 20 \
    --learning_rate $lr \
    --des Exp \
    --lradj type3 \
    --use_multi_scale False \
    --small_kernel_merged False \

done
done
done
done
done