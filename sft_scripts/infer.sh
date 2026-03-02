# If it's full parameter training, use `--model xxx` instead of `--adapters xxx`.
# If you are using the validation set for inference, add the parameter `--load_data_args true`.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters /home/u20249114/ms-swift/svg_output/v12-20260102-163107/checkpoint-3189 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 8192 \
    --val_dataset /home/u20249114/ms-swift/Dataset/test.jsonl \
    --max_batch_size 4
