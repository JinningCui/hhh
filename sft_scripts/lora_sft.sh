# 双卡 DDP 模式 (Native PyTorch Distributed)
# 显存占用: 每张卡约 40GB~60GB (A100 80GB 毫无压力)
# 训练速度: 接近单卡的 1.8 ~ 1.9 倍
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model /home/u20249114/ms-swift/Chart2SVG/models/Qwen3_Chart_Stage1 \
    --train_type lora \
    --dataset '/disk/CJN/Chart2SVG_dataset/medium/train_json/radar' \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --modules_to_save embed_tokens lm_head \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 8192 \
    --truncation_strategy right \
    --output_dir new_svg_output \
    --system 'You are a world-class SVG Expert and Data Visualization Engineer. Your primary objective is to interpret rasterized chart images and reconstruct them into high-quality, semantically correct SVG code.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --attn_impl sdpa \
    --model_author swift \
    --model_name swift-robot \
    --ddp_find_unused_parameters true \
    # --resume_from_checkpoint /home/u20249114/ms-swift/svg_output/v12-20260102-163107/checkpoint-3189