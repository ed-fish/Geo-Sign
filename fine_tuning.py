# finetuning.py

# Geo-Sign
# ---------------------------------------------------------------------------------

# --- Standard Imports ---
import os
import time
import argparse
import json
import datetime
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

# --- PyTorch Imports ---
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
# from torch.nn.parallel import DistributedDataParallel # Not used if DeepSpeed handles DDP

# --- Third-party Imports ---
import numpy as np
from timm.optim import create_optimizer
from transformers import get_scheduler

# --- WandB Logging ---
try:
    import wandb
    _wandb_available = True
except ImportError:
    wandb = None
    _wandb_available = False

# --- Geoopt for Hyperbolic ---
try:
    from geoopt import ManifoldParameter
    from geoopt.optim import RiemannianAdam
    _geoopt_available = True
except ImportError:
    ManifoldParameter = object # type: ignore
    RiemannianAdam = None # type: ignore
    _geoopt_available = False
    # Warning for geoopt not being available is typically handled in models.py

# --- Project-specific Imports ---
from models import Uni_Sign, get_requires_grad_dict # Assumes Uni_Sign is updated as per "simpler approach"
import utils as utils
from datasets import S2T_Dataset
from SLRT_metrics import translation_performance, islr_performance, wer_list
from config import train_label_paths, dev_label_paths, test_label_paths, mt5_path

# ==============================================================================
#                                Main Function
# ==============================================================================
def main(args):
    """Main function orchestrating training and evaluation."""
    utils.init_distributed_mode_ds(args)
    device = torch.device(args.gpu)
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    print(f"Initialized process with rank {rank} on device {device}. World size: {world_size}")
    utils.set_seed(args.seed + rank)

    args.wandb_run = None
    if rank == 0 and hasattr(args, 'wandb') and args.wandb:
        if not _wandb_available:
            warnings.warn("WandB flag is set (--wandb) but the package is not installed. Logging disabled.")
            args.wandb = False
        else:
            try:
                run_name = getattr(args, 'wandb_run_name', None) or \
                           f"{args.dataset}_{args.task}-hyp_{args.use_hyperbolic}-{time.strftime('%Y%m%d-%H%M')}"
                args.wandb_run = wandb.init(
                    project=getattr(args, 'wandb_project', "hyper-sign"),
                    name=run_name,
                    config=vars(args),
                    resume="allow",
                    mode='online'
                )
                print(f"WandB initialized for run: {wandb.run.name} (ID: {wandb.run.id})")
            except Exception as e:
                print(f"WandB initialization failed: {e}")
                args.wandb = False

    if rank == 0: # Print once
        print("Full args:", args)
        if hasattr(args, 'init_c'):
             print("CLI init_c =", args.init_c)

    print("Creating datasets...")
    train_data = S2T_Dataset(path=train_label_paths[args.dataset], args=args, phase='train')
    dev_data = S2T_Dataset(path=dev_label_paths[args.dataset], args=args, phase='dev')
    test_data = S2T_Dataset(path=test_label_paths[args.dataset], args=args, phase='test')
    if rank == 0:
        print(f"Train dataset size: {len(train_data)}")
        print(f"Dev dataset size: {len(dev_data)}")
        print(f"Test dataset size: {len(test_data)}")

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dev_sampler = SequentialSampler(dev_data) # Dev/test typically don't need distributed sampling if eval is on rank 0
    test_sampler = SequentialSampler(test_data)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                  collate_fn=train_data.collate_fn, sampler=train_sampler,
                                  pin_memory=args.pin_mem, drop_last=True)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn, sampler=dev_sampler,
                                pin_memory=args.pin_mem)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn, sampler=test_sampler,
                                 pin_memory=args.pin_mem)
    if rank == 0: print("Dataloaders created.")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if not hasattr(args, 'epochs') or args.epochs <= 0:
        raise ValueError("args.epochs must be provided (> 0) via the argument parser.")
    args.total_steps = num_update_steps_per_epoch * args.epochs
    if rank == 0:
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Steps per epoch: {num_update_steps_per_epoch}")
        print(f"Total training steps calculated: {args.total_steps}")

    print("Creating Uni-Sign model...")
    model = Uni_Sign(args=args) # args (including args.eval) is passed to the model
    model.to(device)

    if hasattr(args, 'finetune') and args.finetune:
        if rank == 0:
            print('***********************************')
            print(f'Loading Model Checkpoint for Finetuning: {args.finetune}')
            print('***********************************')
        if not Path(args.finetune).exists():
            warnings.warn(f"Finetune checkpoint path not found: {args.finetune}. Skipping loading.")
        else:
            try:
                checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('model', checkpoint)
                if not isinstance(state_dict, dict):
                    raise ValueError("Checkpoint does not contain a valid state_dict.")
                
                if not args.eval: # Only strip curvature if not in eval (might want to load it for eval)
                    for k_orig in list(state_dict.keys()):
                        if k_orig.endswith("manifold.isp_c") or k_orig.endswith("manifold.c"): # Check both for safety
                            if rank == 0: # Print only on main process
                                print(f"  » Skipping manifold curvature parameter during finetune load: {k_orig}")
                            state_dict.pop(k_orig)
                
                ret = model.load_state_dict(state_dict, strict=False)
                if rank == 0:
                    if ret.missing_keys: print('Missing keys during finetune load: \n', '\n'.join(ret.missing_keys))
                    if ret.unexpected_keys: print('Unexpected keys during finetune load: \n', '\n'.join(ret.unexpected_keys))
            except Exception as e:
                print(f"ERROR loading finetuning checkpoint: {e}. Model weights remain initialized.")
    elif rank == 0:
        print("No finetuning checkpoint specified (--finetune). Model starts from scratch or pre-trained mT5.")

    model_without_ddp = model
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    if rank == 0: print(f'Model created/loaded. Number of parameters: {n_parameters:.2f}M')

    optimizer = None
    hyp_optimizer = None
    euclid_params = []
    hyp_params_list = [] 

    if args.use_hyperbolic and _geoopt_available:
        if rank == 0: print("Hyperbolic branch is active. Setting up Riemannian and Euclidean optimizers...")
        for name, p in model_without_ddp.named_parameters():
            if not p.requires_grad: continue
            if isinstance(p, ManifoldParameter) or name.endswith("manifold.c"): # Check for manifold.c too
                hyp_params_list.append(p)
            else:
                euclid_params.append(p)
        
        if not hyp_params_list:
            warnings.warn("use_hyperbolic=True but no hyperbolic parameters (ManifoldParameter or 'manifold.c') found. Falling back to standard optimizer.")
            args.use_hyperbolic = False 
            optimizer = create_optimizer(args, model_without_ddp)
        else:
            if rank == 0: print(f"Found {len(hyp_params_list)} hyperbolic and {len(euclid_params)} Euclidean parameters.")
            optimizer = torch.optim.AdamW(
                euclid_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=getattr(args, 'opt_betas', (0.9, 0.98)) 
            )
            hyp_optimizer = RiemannianAdam(
                hyp_params_list,
                lr=args.hyp_lr,
                stabilize=getattr(args, 'hyp_stabilize', True), 
                weight_decay=0.0 
            )
            if rank == 0: 
                print(f"Euclidean Optimizer: {optimizer}")
                print(f"Hyperbolic Optimizer: {hyp_optimizer}")
            model_without_ddp.hyp_optimizer = hyp_optimizer 
    else:
        if args.use_hyperbolic and not _geoopt_available:
            warnings.warn("use_hyperbolic=True but geoopt is not available. Falling back to standard optimizer.")
        elif rank == 0:
             print("Hyperbolic branch inactive or geoopt unavailable. Setting up standard Euclidean optimizer...")
        args.use_hyperbolic = False 
        optimizer = create_optimizer(args, model_without_ddp)
        if rank == 0: print(f"Standard Optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        name=getattr(args, 'scheduler', 'cosine'),
        optimizer=optimizer,
        num_warmup_steps=int(getattr(args, 'warmup_epochs', 0) * num_update_steps_per_epoch),
        num_training_steps=args.total_steps,
    )
    if rank == 0: print(f"LR Scheduler ({getattr(args, 'scheduler', 'cosine')}) configured for main optimizer.")

    if rank == 0: print("Initializing DeepSpeed...")
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    
    # Ensure model_without_ddp points to the raw model after DeepSpeed/DDP wrapping
    _model_to_unwrap = model
    while hasattr(_model_to_unwrap, 'module'):
        _model_to_unwrap = _model_to_unwrap.module
    model_without_ddp = _model_to_unwrap

    if rank == 0: print("DeepSpeed initialized.")

    start_epoch = 0 
    if hasattr(args, 'load_checkpoint_dir') and args.load_checkpoint_dir:
        warnings.warn("--load_checkpoint_dir specified but script uses --finetune for pre-DS loading. DS checkpoint loading ignored.")

    output_dir = Path(args.output_dir) if args.output_dir else None
    start_time = time.time()
    max_accuracy = 0.0
    if args.task == "CSLR": max_accuracy = 1000.0

    if args.eval: 
        if rank == 0: print("Evaluation mode enabled. Running evaluation...")
        if args.task != "ISLR":
            if rank == 0: print("--- Evaluating on Dev Set ---")
            evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
        if rank == 0: print("--- Evaluating on Test Set ---")
        evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
        if args.wandb and rank == 0 and args.wandb_run: args.wandb_run.finish()
        return

    print(f"[Rank {rank}] Starting training from epoch {start_epoch} to {args.epochs-1}")
    for epoch in range(start_epoch, args.epochs):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        if rank == 0: print(f"--- Epoch {epoch}/{args.epochs-1} ---")

        train_stats = train_one_epoch(
            args=args, model=model, data_loader=train_dataloader, optimizer=optimizer,
            hyp_optimizer=hyp_optimizer, 
            epoch=epoch, lr_scheduler=lr_scheduler, model_without_ddp=model_without_ddp
        )
        if rank == 0: print(f"Epoch {epoch} training finished. Avg Loss: {train_stats.get('loss', -1):.4f}")

        if output_dir and rank == 0: 
            checkpoint_path = output_dir / f'checkpoint_{epoch}.pth'
            model_state_to_save = model_without_ddp.state_dict()
            save_payload = {
                'model': model_state_to_save, 'epoch': epoch, 'args': vars(args),
                'max_accuracy': max_accuracy,
                'global_step': model_without_ddp.global_step.item() if hasattr(model_without_ddp, 'global_step') else 0
            }
            if hyp_optimizer: 
                save_payload['hyp_optimizer'] = hyp_optimizer.state_dict()
            utils.save_on_master(save_payload, checkpoint_path)

        print(f"--- Running evaluation for Epoch {epoch} on Rank 0 ---")
        test_stats_dev = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
        test_stats_test = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

        save_best = False
        metric_key, current_metric = "", 0.0
        if args.task == "SLT":
            metric_key, current_metric = "bleu4", test_stats_dev.get("bleu4", 0.0)
            if current_metric > max_accuracy: save_best = True
        elif args.task == "ISLR":
            metric_key, current_metric = "top1_acc_pi", test_stats_dev.get("top1_acc_pi", 0.0)
            if current_metric > max_accuracy: save_best = True
        elif args.task == "CSLR":
            metric_key, current_metric = "wer", test_stats_dev.get("wer", 1000.0)
            if current_metric < max_accuracy: save_best = True
        
        if save_best:
            print(f"*** New best {metric_key}: {current_metric:.2f} (Epoch {epoch}) ***")
            max_accuracy = current_metric
            if output_dir:
                best_checkpoint_path = output_dir / 'best_checkpoint.pth'
                model_state_to_save_best = model_without_ddp.state_dict()
                best_payload = {
                    'model': model_state_to_save_best, 'epoch': epoch, 'args': vars(args),
                    f'best_{metric_key}': max_accuracy,
                    'global_step': model_without_ddp.global_step.item() if hasattr(model_without_ddp, 'global_step') else 0
                }
                if hyp_optimizer:
                     best_payload['hyp_optimizer'] = hyp_optimizer.state_dict()
                utils.save_on_master(best_payload, best_checkpoint_path)

        print(f'Current best {metric_key}: {max_accuracy:.2f}')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'dev_{k}': v for k, v in test_stats_dev.items()},
                     **{f'test_{k}': v for k, v in test_stats_test.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}
        if output_dir:
            try:
                with (output_dir / "log.txt").open("a") as f: f.write(json.dumps(log_stats) + "\n")
            except IOError as e: print(f"[Rank 0] Error writing to log.txt: {e}")
        
        if args.wandb and args.wandb_run:
            wandb_epoch_log = {f"epoch_train_avg/{k}": v for k,v in train_stats.items()}
            wandb_epoch_log.update({f"epoch_dev_avg/{k}": v for k,v in test_stats_dev.items()})
            wandb_epoch_log.update({f"epoch_test_avg/{k}": v for k,v in test_stats_test.items()})
            wandb_epoch_log["epoch"] = epoch
            current_global_step_val = model_without_ddp.global_step.item() if hasattr(model_without_ddp, 'global_step') else (epoch + 1) * num_update_steps_per_epoch
            args.wandb_run.log(wandb_epoch_log, step=int(current_global_step_val))

        if world_size > 1 and utils.dist.is_initialized(): utils.dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        print('Training completed in {}'.format(total_time_str))
        if args.wandb and args.wandb_run:
            args.wandb_run.finish()
            print("WandB run finished.")

# ==============================================================================
#                           Training Epoch Function
# ==============================================================================
def train_one_epoch(args: argparse.Namespace,
                    model: torch.nn.Module,
                    data_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    hyp_optimizer: Optional[torch.optim.Optimizer] = None,
                    model_without_ddp: Optional[Uni_Sign] = None
                   ) -> Dict[str, float]:
    model.train()
    device = args.gpu 
    rank = utils.get_rank()

    if model_without_ddp is None: 
        _model_to_unwrap = model
        while hasattr(_model_to_unwrap, 'module'):
            _model_to_unwrap = _model_to_unwrap.module
        model_without_ddp = _model_to_unwrap

    use_hyp_in_model_runtime = args.use_hyperbolic and \
                               hasattr(model_without_ddp, 'use_hyp') and \
                               model_without_ddp.use_hyp and \
                               hyp_optimizer is not None


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if use_hyp_in_model_runtime :
        metric_logger.add_meter('hyp_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('ce_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    if use_hyp_in_model_runtime:
        metric_logger.add_meter('margin_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('alpha', utils.SmoothedValue(window_size=50, fmt='{value:.3f}'))
        metric_logger.add_meter('curvature', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = f'Epoch: [{epoch}/{args.epochs-1}]'
    print_freq = max(1, len(data_loader) // 10 if len(data_loader) > 10 else 1)

    model.zero_grad() 
    if hyp_optimizer:
        hyp_optimizer.zero_grad(set_to_none=True)

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key in src_input:
            if isinstance(src_input[key], torch.Tensor): src_input[key] = src_input[key].to(device, non_blocking=True)
        for key in tgt_input:
            if isinstance(tgt_input[key], torch.Tensor): tgt_input[key] = tgt_input[key].to(device, non_blocking=True)

        target_dtype = torch.bfloat16 if (hasattr(model, 'bfloat16_enabled') and model.bfloat16_enabled()) or \
                                         (hasattr(args, 'bf16') and args.bf16) else None
        if target_dtype:
            long_keys = {'prefix_ids', 'labels_ids'}
            for key, val in src_input.items():
                if isinstance(val, torch.Tensor):
                    src_input[key] = val.to(torch.long if key in long_keys else target_dtype, non_blocking=True)
            for key, val in tgt_input.items():
                 if isinstance(val, torch.Tensor) and key == 'labels_ids': 
                      tgt_input[key] = val.to(torch.long, non_blocking=True)

        if args.task == "CSLR": 
            if 'gt_gloss' in tgt_input: tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            # else: handle missing key if necessary, though dataset should provide it

        stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']

        model.backward(total_loss) 

        grad_norm_hyp = None
        if hasattr(args, 'manual_grad_clip') and args.manual_grad_clip and use_hyp_in_model_runtime and hyp_optimizer:
            hyp_params_list_runtime = [p for p_group in hyp_optimizer.param_groups for p in p_group['params'] if p.requires_grad]
            if hyp_params_list_runtime and hasattr(args, 'clip_grad_norm_hyp') and args.clip_grad_norm_hyp > 0:
                valid_hyp_grads = [p for p in hyp_params_list_runtime if p.grad is not None]
                if valid_hyp_grads:
                    grad_norm_hyp = clip_grad_norm_(valid_hyp_grads, max_norm=args.clip_grad_norm_hyp)
        
        if hyp_optimizer: hyp_optimizer.step() 
        model.step() 
        
        if hyp_optimizer: hyp_optimizer.zero_grad(set_to_none=True)

        if use_hyp_in_model_runtime and hasattr(model_without_ddp, 'global_step'):
            model_without_ddp.global_step += 1 # type: ignore
        
        current_global_step_val = model_without_ddp.global_step.item() if use_hyp_in_model_runtime and hasattr(model_without_ddp, 'global_step') \
                                  else ((epoch * len(data_loader) + step) // args.gradient_accumulation_steps)

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"[Rank {rank}] Loss is {loss_value} at step {current_global_step_val}, stopping training")
            if utils.get_world_size() > 1 and utils.dist.is_initialized(): utils.dist.barrier()
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(ce_loss=stack_out.get('ce_loss', torch.tensor(0.0)).item())
        current_lr_val = optimizer.param_groups[0]['lr'] 
        metric_logger.update(lr=current_lr_val)
        
        if use_hyp_in_model_runtime and hyp_optimizer:
            metric_logger.update(hyp_lr=hyp_optimizer.param_groups[0]["lr"])
            metric_logger.update(margin_loss=stack_out.get('margin_loss', torch.tensor(0.0)).item())
            metric_logger.update(alpha=stack_out.get('alpha', torch.tensor(0.0)).item())
            metric_logger.update(curvature=stack_out.get('curvature', torch.tensor(0.0)).item())

        if args.wandb and rank == 0 and args.wandb_run and (current_global_step_val % print_freq == 0 or step == len(data_loader) -1) :
            wandb_log = {
                'batch_loss': total_loss.item(),
                'batch_ce_loss': stack_out.get('ce_loss', torch.tensor(0.0)).item(),
                'lr': current_lr_val,
            }
            if use_hyp_in_model_runtime and hyp_optimizer:
                wandb_log.update({
                    'batch_margin_loss': stack_out.get('margin_loss', torch.tensor(0.0)).item(),
                    'batch_alpha': stack_out.get('alpha', torch.tensor(0.0)).item(),
                    'geometry/curvature': stack_out.get('curvature', torch.tensor(0.0)).item(),
                    'weights/body': stack_out.get('weights_fm_body', torch.tensor(0.0)).item(),
                    'weights/left': stack_out.get('weights_fm_left', torch.tensor(0.0)).item(),
                    'weights/right': stack_out.get('weights_fm_right', torch.tensor(0.0)).item(),
                    'weights/face': stack_out.get('weights_fm_face', torch.tensor(0.0)).item(),
                    'hyp_sim_mean': stack_out.get('hyp_sim_mean', torch.tensor(0.0)).item(),
                    'effective_margin': stack_out.get('effective_margin', torch.tensor(0.0)).item(),
                    'temperature': stack_out.get('temperature', torch.tensor(0.0)).item(),
                    'lr_hyperbolic': hyp_optimizer.param_groups[0]['lr']
                })
                if grad_norm_hyp is not None:
                    wandb_log['grad_norm/hyperbolic'] = grad_norm_hyp.item()
            args.wandb_run.log(wandb_log, step=int(current_global_step_val))

        if hasattr(args, 'quick_break') and args.quick_break > 0 and step >= (args.quick_break -1):
            print(f"[Rank {rank}] Reached quick_break step {args.quick_break}, ending epoch early.")
            break
            
    metric_logger.synchronize_between_processes()
    if rank == 0: print(f"Epoch {epoch} averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ==============================================================================
#                           Evaluation Function
# ==============================================================================
@torch.no_grad()
def evaluate(args: argparse.Namespace,
             data_loader: DataLoader,
             model: torch.nn.Module,
             model_without_ddp: Uni_Sign, 
             phase: str
            ) -> Dict[str, float]:
    model.eval()
    device = args.gpu 
    rank = utils.get_rank()
    world_size = utils.get_world_size()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
    if args.task == "SLT":
        for i in range(1, 5): metric_logger.add_meter(f'bleu{i}', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('rouge', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    elif args.task == "ISLR":
        metric_logger.add_meter('top1_acc_pi', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('top1_acc_pc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif args.task == "CSLR":
        metric_logger.add_meter('wer', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('sub', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('ins', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('del', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    use_hyp_in_model_eval = args.use_hyperbolic and \
                            hasattr(model_without_ddp, 'use_hyp') and \
                            model_without_ddp.use_hyp
    if use_hyp_in_model_eval:
        metric_logger.add_meter('hyp_sim_mean', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
        metric_logger.add_meter('curvature', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))

    header = f'Eval ({phase}):'
    print_freq = max(1, len(data_loader) // 10 if len(data_loader) > 10 else 1)

    tgt_pres_text_local: List[str] = []
    tgt_refs_text_local: List[str] = []

    collected_figure_data_first_batch: Optional[Dict[str, Any]] = None
    save_one_batch_flag: bool = getattr(args, 'save_one_batch', False)
    save_batch_name_from_args: str = getattr(args, 'save_batch_name', f"{phase}_first_batch_fig_data_DEFAULTNAME")

    accumulated_eval_samples_for_phase: List[Dict[str, torch.Tensor]] = []
    max_samples_to_store_for_viz = getattr(args, "max_eval_samples", 500)

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key in src_input:
            if isinstance(src_input[key], torch.Tensor): src_input[key] = src_input[key].to(device, non_blocking=True)
        for key in tgt_input:
            if isinstance(tgt_input[key], torch.Tensor): tgt_input[key] = tgt_input[key].to(device, non_blocking=True)

        target_dtype = torch.bfloat16 if (hasattr(model, 'bfloat16_enabled') and model.bfloat16_enabled()) or \
                                         (hasattr(args, 'bf16') and args.bf16) else None
        if target_dtype:
            long_keys = {'prefix_ids', 'labels_ids'}
            for key, val in src_input.items():
                if isinstance(val, torch.Tensor):
                    src_input[key] = val.to(torch.long if key in long_keys else target_dtype, non_blocking=True)
            for key, val in tgt_input.items():
                 if isinstance(val, torch.Tensor) and key == 'labels_ids':
                      tgt_input[key] = val.to(torch.long, non_blocking=True)
        
        if args.task == "CSLR":
            if 'gt_gloss' in tgt_input: tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            elif 'gt_sentence' not in tgt_input:
                warnings.warn(f"Missing 'gt_gloss' or 'gt_sentence' for CSLR task in batch {step}. Using empty references.")
                bs = src_input.get(next(iter(src_input)), torch.tensor([])).shape[0] # type: ignore
                tgt_input['gt_sentence'] = [""] * bs

        stack_out = model(src_input, tgt_input) 

        if 'loss' in stack_out and torch.is_tensor(stack_out['loss']):
            metric_logger.update(loss=stack_out['loss'].item())
        if use_hyp_in_model_eval:
            if 'hyp_sim_mean' in stack_out and torch.is_tensor(stack_out['hyp_sim_mean']):
                 metric_logger.update(hyp_sim_mean=stack_out['hyp_sim_mean'].item())
            if 'curvature' in stack_out and torch.is_tensor(stack_out['curvature']):
                 metric_logger.update(curvature=stack_out['curvature'].item())
        
        batch_eval_data = stack_out.get("eval_figure_data", {})

        if rank == 0 and step < 3: # Debug print for first few steps
            print(f"[DEBUG evaluate] Step {step}, Phase {phase}, Global args.eval: {args.eval}, batch_eval_data keys: {list(batch_eval_data.keys())}")

        if rank == 0 and save_one_batch_flag and batch_eval_data: 
            if collected_figure_data_first_batch is None:
                try:
                    current_first_batch_to_save = batch_eval_data.copy() 
                    if 'gt_sentence' in tgt_input:
                        current_first_batch_to_save["references_raw"] = list(tgt_input['gt_sentence'])
                    current_first_batch_to_save["batch_index"] = step
                    collected_figure_data_first_batch = current_first_batch_to_save 

                    if args.output_dir:
                        figure_data_path = Path(args.output_dir) / f'{save_batch_name_from_args}.pt'
                        torch.save(current_first_batch_to_save, figure_data_path)
                        print(f"\n[Rank 0] Saved single batch figure data from batch {step} to {figure_data_path}")
                    else:
                        warnings.warn("[Rank 0] Output dir not specified. Cannot save single batch figure data.", RuntimeWarning)
                except Exception as e:
                    warnings.warn(f"[Rank 0] Error saving single batch figure data: {e}", RuntimeWarning)
                    collected_figure_data_first_batch = "ERROR" 

        if args.eval and batch_eval_data: 
            if len(accumulated_eval_samples_for_phase) < max_samples_to_store_for_viz:
                accumulated_eval_samples_for_phase.append(batch_eval_data) 
                if rank == 0 and len(accumulated_eval_samples_for_phase) % 50 == 0 : 
                     print(f"[DEBUG evaluate] Step {step}, Phase {phase}, Accumulated {len(accumulated_eval_samples_for_phase)} samples for visualization.")

        try:
            generation_input_payload = {
                 "inputs_embeds": stack_out["inputs_embeds"].to(device), 
                 "attention_mask": stack_out["attention_mask"].to(device)
            }
            output_ids = model_without_ddp.generate(
                pc=generation_input_payload,
                max_new_tokens=getattr(args, 'max_tgt_len', 100),
                num_beams=getattr(args, 'num_beams', 4),
                **getattr(args, 'generation_kwargs', {})
            )
        except Exception as e:
            print(f"\n[Rank {rank}] Error during generation at step {step}, phase {phase}: {e}")
            bs = src_input.get(next(iter(src_input)), torch.tensor([])).shape[0] # type: ignore
            output_ids = torch.zeros((bs, 1), dtype=torch.long, device=device)

        try:
            tokenizer = model_without_ddp.mt5_tokenizer
            decoded_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            tgt_pres_text_local.extend(decoded_preds)
            if 'gt_sentence' in tgt_input:
                tgt_refs_text_local.extend(list(tgt_input['gt_sentence']))
            else: 
                tgt_refs_text_local.extend([""] * len(decoded_preds))
        except Exception as e:
            print(f"\n[Rank {rank}] Error during decoding/storing texts at step {step}, phase {phase}: {e}")
            bs = src_input.get(next(iter(src_input)), torch.tensor([])).shape[0] # type: ignore
            tgt_pres_text_local.extend(["DECODING_ERROR"] * bs)
            tgt_refs_text_local.extend([""] * bs)

    if world_size > 1 and utils.dist.is_initialized(): utils.dist.barrier()

    gathered_preds_obj: List[Optional[List[str]]] = [None] * world_size
    gathered_refs_obj: List[Optional[List[str]]] = [None] * world_size
    if world_size > 1 and utils.dist.is_initialized():
        utils.dist.all_gather_object(gathered_preds_obj, tgt_pres_text_local)
        utils.dist.all_gather_object(gathered_refs_obj, tgt_refs_text_local)
    else:
        gathered_preds_obj[0] = tgt_pres_text_local
        gathered_refs_obj[0] = tgt_refs_text_local
    
    final_metrics_dict = {}

    if rank == 0:
        all_preds = [item for sublist in gathered_preds_obj if sublist is not None for item in sublist]
        all_refs = [item for sublist in gathered_refs_obj if sublist is not None for item in sublist]
        print(f"\nEvaluation ({phase}) gathered {len(all_preds)} predictions and {len(all_refs)} references.")

        formatted_pres, formatted_refs = [], []
        if not all_preds or not all_refs or len(all_preds) != len(all_refs):
            warnings.warn(f"Pred/Ref mismatch for {phase} ({len(all_preds)} vs {len(all_refs)}). Metrics may be off.")
        else:
            if args.dataset == 'CSL_Daily' and args.task == "SLT":
                formatted_pres = [' '.join(list(r.replace(" ","").replace("\n",""))) for r in all_preds]
                formatted_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",""))) for r in all_refs]
            elif args.dataset == 'CSL_News':
                formatted_pres = [' '.join(list(r.replace(" ", "").replace("\n", ""))) for r in all_preds]
                formatted_refs = [' '.join(list(r.replace("，", ",").replace("？", "?").replace(" ", ""))) for r in all_refs]
            else:
                formatted_pres, formatted_refs = all_preds, all_refs

            try: 
                if args.task == "SLT":
                    bleu_dict, rouge_score = translation_performance(formatted_refs, formatted_pres)
                    for k,v in bleu_dict.items():
                        if k in metric_logger.meters: metric_logger.meters[k].update(v)
                    if 'rouge' in metric_logger.meters: metric_logger.meters['rouge'].update(rouge_score)
                elif args.task == "ISLR":
                    top1_acc_pi, top1_acc_pc = islr_performance(formatted_refs, formatted_pres) 
                    if 'top1_acc_pi' in metric_logger.meters: metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
                    if 'top1_acc_pc' in metric_logger.meters: metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
                elif args.task == "CSLR":
                    wer_results = wer_list(hypotheses=formatted_pres, references=formatted_refs) 
                    for k,v in wer_results.items():
                        if k in metric_logger.meters: metric_logger.meters[k].update(v)
            except Exception as e: print(f"\n[Rank 0] ERROR calculating {phase} metrics: {e}")
        
        metric_logger.synchronize_between_processes() 
        print(f'\n* Evaluation Complete ({phase} - Rank 0 Results):')
        for name, meter in metric_logger.meters.items(): print(f"  {name}: {meter.global_avg:.4f}")

        if args.eval and args.output_dir and formatted_pres and formatted_refs: 
            try:
                pred_p = Path(args.output_dir) / f'{phase}_eval_predictions.txt'
                ref_p = Path(args.output_dir) / f'{phase}_eval_references.txt'
                with open(pred_p, 'w', encoding='utf-8') as f:
                    for line in formatted_pres: f.write(line + '\n')
                with open(ref_p, 'w', encoding='utf-8') as f:
                    for line in formatted_refs: f.write(line + '\n')
                print(f"[Rank 0] Saved {phase} predictions to {pred_p} and references to {ref_p}")
            except IOError as e: print(f"[Rank 0] ERROR saving {phase} eval text files: {e}")

        # --- Save Accumulated Evaluation Figure Data (Rank 0 Only) ---
        if args.eval and accumulated_eval_samples_for_phase: 
            if args.output_dir:
                num_s = len(accumulated_eval_samples_for_phase)
                accum_fname = f"{phase}_all_eval_figure_data_{num_s}_samples.pt"
                accum_fpath = Path(args.output_dir) / accum_fname
                try:
                    print(f"\n[Rank 0] Saving {num_s} accumulated eval figure samples for '{phase}' to {accum_fpath}...")
                    torch.save(accumulated_eval_samples_for_phase, accum_fpath)
                    print(f"[Rank 0] Accumulated eval figure data saved.")
                except Exception as e: warnings.warn(f"[Rank 0] Error saving accumulated eval data for '{phase}': {e}", RuntimeWarning)
            else: warnings.warn(f"[Rank 0] Output dir not specified. Skipping save of accumulated eval data for '{phase}'.", RuntimeWarning)
        elif rank == 0 and args.eval: 
            print(f"\n[Rank 0] No accumulated eval figure data to save for phase '{phase}' (list is empty or global --eval was false). List length: {len(accumulated_eval_samples_for_phase)}")


        if args.wandb and args.wandb_run:
            try:
                step_val = model_without_ddp.global_step.item() if hasattr(model_without_ddp, 'global_step') else args.epochs * len(data_loader)
                wandb_m = {f"{phase}_final/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
                args.wandb_run.log(wandb_m, step=int(step_val))
            except Exception as e: print(f"[Rank 0] Error logging final eval metrics to WandB: {e}")
    
    metric_logger.synchronize_between_processes() 
    final_metrics_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return final_metrics_dict

# ==============================================================================
#                           Script Entry Point
# ==============================================================================
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    cli_args = parser.parse_args() 

    if utils.is_main_process() and cli_args.output_dir:
        Path(cli_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(cli_args)
    print("Script finished.")
