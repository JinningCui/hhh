export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_TIMEOUT=22
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_RETRY_CNT=13
export NCCL_BLOCKING_WAIT=1
# 这是一个关键：强制设置分布式同步超时为 3600 秒
export DISTRIBUTED_TIMEOUT=3600

MAX_PIXELS=262144 \
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
  --rlhf_type grpo \
  --model /home/u20249114/ms-swift/Chart2SVG/models/Qwen3_Chart_Stage1 \
  --train_type lora \
  --model_type qwen3_vl \
  --dataset '/home/u20249114/ms-swift/Dataset/Beagle/grpo_train_800' \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --external_plugins /disk/CJN/ms-swift/ms-swift-release-3.12/examples/train/grpo/plugin/plugin.py \
  --reward_funcs svg_pipeline \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_length 8192 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_steps 10 \
  --save_steps 10 \
  --learning_rate 1e-5 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 0 \
  --max_completion_length 8192 \
  --num_generations 4 \
  --use_vllm false \
  --deepspeed zero3 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 80 \
  --output_dir output_grpo \
  --logging_dir logging_grpo \
  --log_completions true \
  --report_to tensorboard \
  --beta 0.005 \
  --max_grad_norm 0.5 \
  --num_iterations 1 \
