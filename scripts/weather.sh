for model_name in ModernTCN Transformer Informer Reformer Flowformer \
    Flashformer iTransformer iInformer iReformer iFlowformer iFlashformer
do
for batch_size in 128 256 512
do
for lr in 0.0001
do
for seq_len in 96 192 336
do
for pred_len in 96 192 336 720
do

python -u run.py \
    --is_training 1 \
    --model_id weather \
    --model $model_name \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --log_dir ./logs/ \
    --log_name weather.txt \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --ffn_ratio 8 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 1 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.0 \
    --enc_in 21 \
    --dec_in 21 \
    --e_layers 3 \
    --c_out 21 \
    --d_model 512 \
    --d_ff 512 \
    --dropout 0.4 \
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