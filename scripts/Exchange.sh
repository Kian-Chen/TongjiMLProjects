for model_name in ModernTCN Transformer Informer Reformer Flowformer \
    Flashformer iTransformer iInformer iReformer iFlowformer iFlashformer
do
for batch_size in 128 256 512
do
for lr in 0.0001
do
for seq_len in 6 12 24 36 72 96
do
for pred_len in 96 192 336 720
do

python -u run.py \
    --is_training 1 \
    --model_id Exchange \
    --model $model_name \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --log_dir ./logs/ \
    --log_name Exchange.txt \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --ffn_ratio 1 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.6 \
    --enc_in 8 \
    --dec_in 8 \
    --e_layers 2 \
    --c_out 8 \
    --d_model 128 \
    --d_ff 128 \
    --dropout 0.2 \
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