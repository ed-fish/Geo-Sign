output_dir=out/stage3_train

# RGB-pose setting
ckpt_path=checkpoints/best.pth


deepspeed fine_tuning.py \
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
  --save_one_batch \
  --eval

  
#   # --rgb_support # enable RGB-pose setting

# example of ISLR
# deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
#    --batch-size 8 \
#    --gradient-accumulation-steps 1 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --output_dir $output_dir \
#    --finetune $ckpt_path \
#    --dataset WLASL \
#    --task ISLR \
#    --use_hyperbolic \
#    --max_length 64 \
  #  --rgb_support # enable RGB-pose setting

## pose only setting
#ckpt_path=out/stage1_pretraining/best_checkpoint.pth
#
#deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
#  --batch-size 8 \
#  --gradient-accumulation-steps 1 \
#  --epochs 20 \
#  --opt AdamW \
#  --lr 3e-4 \
#  --output_dir $output_dir \
#  --finetune $ckpt_path \
#  --dataset CSL_Daily \
#  --task SLT \
##   --rgb_support # enable RGB-pose setting
