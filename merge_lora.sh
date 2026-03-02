#!/bin/bash

# Merging LoRA weights into the base model
# Source Checkpoint: v12 (Step 3189)
# Target Model Directory: Qwen3_Chart_Stage1

echo "Start merging LoRA weights..."
echo "Source: /home/u20249114/ms-swift/svg_output/v12-20260102-163107/checkpoint-3189"
echo "Target: /home/u20249114/ms-swift/Chart2SVG/models/Qwen3_Chart_Stage1"

CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir /home/u20249114/ms-swift/svg_output/v12-20260102-163107/checkpoint-3189 \
    --merge_lora true \
    --output_dir /home/u20249114/ms-swift/Chart2SVG/models/Qwen3_Chart_Stage1

echo "Merge completed!"
