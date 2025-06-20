output_dir=out/stage3_train

# RGB-pose setting
ckpt_path=checkpoints/pretraining.pth

deepspeed --num_gpus 1 fine_tuning.py \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs 40 \
  --opt AdamW \
  --lr 1e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task SLT \
  --use_hyperbolic \
  --wandb \
  --init_c 1.0 \
  --hyp_text_emb_src decoder \
  --hyp_text_cmp token \
  --alpha 0.7 \
