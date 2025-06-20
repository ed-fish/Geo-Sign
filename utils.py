"""
This file is modified from:
https://github.com/facebookresearch/deit/blob/main/utils.py

Includes miscellaneous functions, distributed helpers, DeepSpeed config,
and argument parsing for Uni-Sign training.
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import io
import os
import time, random
import numpy as np
from collections import defaultdict, deque
import datetime
import warnings # Added for warnings

import torch
# Use deepspeed comm directly if available and initialized for consistency
# import torch.distributed as dist # Replaced by deepspeed.comm
import torch.nn.functional as F
# from torch import Tensor # Removed Tensor import as it's not used after removing hints
import argparse
import torch.backends.cudnn as cudnn

import pickle
import gzip

# --- DeepSpeed Integration ---
try:
    import deepspeed
    # Use deepspeed's communication library
    import deepspeed.comm as dist
    _deepspeed_available = True
except ImportError:
    warnings.warn("DeepSpeed not installed. Distributed training and ZeRO features will be unavailable. `pip install deepspeed`")
    # Define dummy dist for basic checks to pass if DS not installed
    class DummyDist:
        def is_available(self): return False
        def is_initialized(self): return False
        def get_world_size(self): return 1
        def get_rank(self): return 0
        def barrier(self): pass
        def all_reduce(self, tensor, op=None): pass # op is ignored
        def all_gather_object(self, obj_list, obj): obj_list[0] = obj # Simplistic fallback

    dist = DummyDist()
    _deepspeed_available = False

# Import get_accelerator only if deepspeed is available
if _deepspeed_available:
    from deepspeed.accelerator import get_accelerator
else:
    # Define a dummy get_accelerator if deepspeed is not installed
    class DummyAccelerator:
        def current_device_name(self): return 'cpu' # Or 'cuda' if you want to assume cuda
        def device_count(self): return 1 # Assume 1 device
    def get_accelerator(): return DummyAccelerator()


# ==============================================================================
#                           SmoothedValue & MetricLogger
# ==============================================================================

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            # Default format includes median and global average
            fmt = "{median:.4f} ({global_avg:.4f})"
        # Use deque for efficient rolling window
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Update the tracked value."""
        # Ensure value is a float or int for calculations
        if isinstance(value, torch.Tensor):
             value = value.item() # Convert tensor to scalar
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque! Only synchronizes total and count.
        """
        if not is_dist_avail_and_initialized():
            return
        # Use the correct communication backend (deepspeed.comm)
        # Ensure device is cuda if available, otherwise cpu
        device = get_accelerator().current_device_name() if torch.cuda.is_available() else 'cpu'
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t) # Use dist from deepspeed.comm
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """Return the median of the values in the window."""
        # Convert deque to tensor for median calculation
        # Handle empty deque case
        if not self.deque: return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Return the average of the values in the window."""
        # Convert deque to tensor for mean calculation
        # Handle empty deque case
        if not self.deque: return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """Return the average over all values added."""
        # Avoid division by zero
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        """Return the maximum value in the window."""
        # Handle empty deque
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        """Return the last value added."""
        # Handle empty deque
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        """String representation using the defined format."""
        # Handle potential errors if properties are accessed on empty deque
        try:
            return self.fmt.format(
                median=self.median,
                avg=self.avg,
                global_avg=self.global_avg,
                max=self.max,
                value=self.value)
        except IndexError: # If deque is empty, value might fail
             return self.fmt.format(
                median=0.0, avg=0.0, global_avg=self.global_avg, max=0.0, value=0.0
             )


class MetricLogger(object):
    """Logs metrics during training/evaluation."""
    def __init__(self, delimiter="\t"):
        # Use defaultdict for easy addition of new meters
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Update meters with new values."""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item() # Convert tensor to scalar
            # Ensure value is suitable for SmoothedValue
            if not isinstance(v, (float, int)):
                 warnings.warn(f"Metric '{k}' has non-numeric type {type(v)}. Skipping update.")
                 continue
            # Update the corresponding meter, creates if not exists
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Allow accessing meters directly via attributes."""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """String representation of all meters."""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronize all meters across distributed processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Add a pre-configured meter."""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Log metrics periodically during iteration."""
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # Determine space format based on iterable length
        try:
             num_iterations = len(iterable)
             space_fmt = ':' + str(len(str(num_iterations))) + 'd'
        except TypeError: # Handle iterables without __len__
             num_iterations = '?'
             space_fmt = ':?d'


        # Base log message format
        log_msg_parts = [
            header,
            '[{0' + space_fmt + '}/{1}]', # Iteration progress
            'eta: {eta}', # Estimated time remaining
            '{meters}', # Meter values
            'time: {time}', # Time per iteration
            'data: {data}' # Data loading time
        ]
        # Add GPU memory usage if available
        if torch.cuda.is_available():
            log_msg_parts.append('max mem: {memory:.0f}MB') # Use MB for readability
        log_msg = self.delimiter.join(log_msg_parts)

        MB = 1024.0 * 1024.0 # Conversion factor for memory logging

        for obj in iterable:
            data_time.update(time.time() - end) # Log data loading time
            yield obj # Yield the item from the iterable
            iter_time.update(time.time() - end) # Log iteration time (including forward/backward)

            # Check if it's time to log
            should_log = (i % print_freq == 0) or (i == num_iterations - 1 if num_iterations != '?' else False)

            if should_log and is_main_process(): # Log only on main process
                # Calculate ETA
                if num_iterations != '?':
                     eta_seconds = iter_time.global_avg * (num_iterations - i)
                     eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                     eta_string = 'N/A' # ETA not available for unknown length iterables

                # Format log message
                format_args = {
                    'meters': str(self),
                    'time': str(iter_time),
                    'data': str(data_time),
                    'eta': eta_string,
                }
                # Add memory info if available
                if torch.cuda.is_available():
                    try: # Add try-except for memory allocation check
                         format_args['memory'] = torch.cuda.max_memory_allocated() / MB
                    except Exception as e:
                         format_args['memory'] = -1 # Indicate error or unavailable
                         # print(f"Warning: Could not get max memory allocated: {e}")
                else:
                     format_args['memory'] = 0 # No GPU memory if not available

                # Print the log message, handling unknown length iterables
                if num_iterations != '?':
                     print(log_msg.format(i, num_iterations, **format_args))
                else:
                     # Simplified format without total iterations
                     simplified_log_msg = self.delimiter.join(log_msg_parts[:1] + log_msg_parts[2:])
                     print(simplified_log_msg.format(i, **format_args))


            i += 1
            end = time.time() # Reset end time for next iteration timing

        # Log total time at the end
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        avg_time_per_it = total_time / num_iterations if num_iterations != '?' and num_iterations > 0 else 0.0
        if is_main_process():
             print('{} Total time: {} ({:.4f} s / it)'.format(
                 header, total_time_str, avg_time_per_it))


# ==============================================================================
#                             Misc Helper Functions
# ==============================================================================

def count_parameters_in_MB(model):
    """Count the total number of parameters in a model in Megabytes."""
    return sum(p.numel() for p in model.parameters()) / 1e6

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """Workaround for ModelEma._load_checkpoint to accept an already-loaded object."""
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file) # Assuming model_ema has this method

def set_seed(seed):
    """Sets random seed for reproducibility across libraries."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups

    np.random.seed(seed)
    random.seed(seed)

    # Configure cuDNN for reproducibility if desired
    # cudnn.deterministic = True
    # cudnn.benchmark = False

def load_dataset_file(filename):
    """Loads a dataset file compressed with gzip and pickled."""
    try:
        with gzip.open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filename}")
        return None
    except Exception as e:
        print(f"Error loading dataset file {filename}: {e}")
        return None

def yield_tokens(file_path):
    """Generator function to yield tokens from a file, line by line."""
    try:
        with io.open(file_path, encoding='utf-8') as f:
            for line in f:
                yield line.strip().split()
    except FileNotFoundError:
        print(f"Error: Token file not found at {file_path}")


# ==============================================================================
#                        Distributed Helper Functions
# ==============================================================================

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # Print only if it's the main process or force=True is passed
        if is_master or force:
            builtin_print(*args, **kwargs)

    # Override the built-in print function
    __builtin__.print = print

def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not _deepspeed_available:
         return False
    try:
         return dist.is_initialized() and dist.get_world_size() > 1
    except Exception:
         return False


def get_world_size():
    """Get the number of distributed processes."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    """Get the rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    """Save data only on the main process."""
    if is_main_process():
        try:
            torch.save(*args, **kwargs)
        except Exception as e:
             print(f"ERROR saving checkpoint on master process: {e}")

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors across all GPUs.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not is_dist_avail_and_initialized():
        return tensor # Return input if not distributed

    world_size = get_world_size()
    # Create a list to hold tensors from all processes
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    # Perform all_gather using deepspeed.comm
    dist.all_gather(tensors_gather, tensor, async_op=False)

    # Concatenate gathered tensors along dimension 0
    output = torch.cat(tensors_gather, dim=0)
    return output

# ==============================================================================
#                        DeepSpeed Helper Functions
# ==============================================================================

def init_distributed_mode_ds(args):
    """Initializes distributed mode using DeepSpeed."""
    if not _deepspeed_available:
         print("DeepSpeed not available, cannot initialize distributed mode via DeepSpeed.")
         args.distributed = False
         args.rank = 0
         args.world_size = 1
         args.gpu = 0 # Assume single GPU
         torch.cuda.set_device(args.gpu) # Set device even for non-distributed
         return

    # Check common environment variables for rank and world size
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        # LOCAL_RANK is crucial for DeepSpeed
        args.gpu = int(os.environ.get('LOCAL_RANK', 0)) # Default to 0 if not set
    elif 'SLURM_PROCID' in os.environ: # Handle SLURM environments
        args.rank = int(os.environ['SLURM_PROCID'])
        # Infer local rank based on available devices
        args.gpu = args.rank % torch.cuda.device_count()
        # Potentially need to set WORLD_SIZE from SLURM variables too
        if 'SLURM_NTASKS' in os.environ:
             args.world_size = int(os.environ['SLURM_NTASKS'])
        else:
             warnings.warn("SLURM environment detected, but WORLD_SIZE could not be inferred from SLURM_NTASKS.")
             args.world_size = 1 # Fallback, might be incorrect
    else:
        # Not a standard distributed environment detected
        print('Distributed environment variables (RANK, WORLD_SIZE, LOCAL_RANK) not found.')
        print('Assuming single GPU / non-distributed mode.')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0 # Assume GPU 0
        torch.cuda.set_device(args.gpu)
        return # Exit early if not distributed

    # If environment variables were found, proceed with DeepSpeed init
    args.distributed = True
    torch.cuda.set_device(args.gpu) # Set the device for this process
    args.dist_backend = 'nccl' # DeepSpeed typically uses NCCL

    print(f'| Distributed init (Rank {args.rank}/{args.world_size}, Local Rank {args.gpu}): Initializing DeepSpeed Process Group')
    # DeepSpeed's init_distributed handles torch.distributed.init_process_group
    deepspeed.init_distributed(dist_backend=args.dist_backend)

    print(f'| Distributed init (Rank {args.rank}): Process group initialized. Waiting at barrier.')
    dist.barrier() # Synchronize all processes after initialization
    print(f'| Distributed init (Rank {args.rank}): Barrier passed. Setting up print suppression.')
    # Suppress printing on non-main processes
    setup_for_distributed(args.rank == 0)
    print(f"| Distributed setup complete for Rank {args.rank}.")


def get_train_ds_config(offload, dtype, stage=2, enable_hybrid_engine=False,
                        inference_tp_size=1, release_inference_cache=False,
                        pin_parameters=True, tp_gather_partition_size=8,
                        max_out_tokens=512, enable_tensorboard=False,
                        enable_mixed_precision_lora=False, tb_path="",
                        tb_name="", args=None):
    """Generates DeepSpeed configuration dictionary for training."""

    device = "cpu" if offload else "none"
    data_type_config_key = None
    dtype_config = {"enabled": False}

    if dtype == "fp16":
        data_type_config_key = "fp16"
        dtype_config = {
            "enabled": True, "loss_scale": 0, "loss_scale_window": 1000,
            "hysteresis": 2, "min_loss_scale": 1
        }
    elif dtype == "bf16":
        data_type_config_key = "bfloat16"
        dtype_config = {"enabled": True}
    elif dtype != "fp32":
         warnings.warn(f"Unsupported dtype '{dtype}' specified for DeepSpeed config. Using fp32.")

    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device, "pin_memory": True},
        "offload_optimizer": {"device": device, "pin_memory": True},
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
        "contiguous_gradients": True,
        "overlap_comm": True
    }

    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.is_initialized() and dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator().device_count()

    ds_config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps * get_world_size(),
        "train_micro_batch_size_per_gpu": args.batch_size,
        "steps_per_print": 10,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "zero_optimization": zero_opt_dict,
        "gradient_clipping": args.gradient_clipping if hasattr(args, 'gradient_clipping') else 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine, "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size, "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters, "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard, "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        },
    }

    if data_type_config_key:
         ds_config[data_type_config_key] = dtype_config

    # Calculate num_update_steps_per_epoch inside if needed, requires train_dataloader length
    # For now, assume total_steps is pre-calculated in args if scheduler is used
    if args and hasattr(args, 'scheduler') and hasattr(args, 'total_steps') and hasattr(args, 'warmup_epochs'):
        # Estimate steps per epoch if possible (requires dataloader length, not available here)
        # Placeholder - assumes total_steps and warmup_num_steps are correctly calculated elsewhere
        warmup_num_steps = getattr(args, 'warmup_num_steps', 0) # Get precalculated value if exists
        total_num_steps = getattr(args, 'total_steps', 0)

        ds_config["scheduler"] = {
            "type": args.scheduler,
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": warmup_num_steps,
                "total_num_steps": total_num_steps
            }
        }

    return ds_config


def init_deepspeed(args, model, optimizer, lr_scheduler):
    """Initializes DeepSpeed engine."""

    if not _deepspeed_available:
         print("DeepSpeed not available. Skipping DeepSpeed initialization.")
         return model, optimizer, lr_scheduler

    ds_config = get_train_ds_config(
        offload=getattr(args, 'offload', False),
        dtype=getattr(args, 'dtype', 'fp32'),
        stage=getattr(args, 'zero_stage', 0),
        args=args
    )

    if is_main_process():
         print("--- Generated DeepSpeed Config ---")
         try: print(json.dumps(ds_config, indent=2))
         except Exception as e: print(f"Could not print DeepSpeed config: {e}")
         print("---------------------------------")

    print(f"[Rank {get_rank()}] Initializing DeepSpeed engine...")
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters())) if optimizer is None else None

    model_engine, optimizer_engine, _, lr_scheduler_engine = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=ds_config,
        dist_init_required=False
    )
    print(f"[Rank {get_rank()}] DeepSpeed engine initialized.")

    return model_engine, optimizer_engine, lr_scheduler_engine


# ==============================================================================
#                         Argument Parser Setup
# ==============================================================================
def get_args_parser():
    """Defines the base argument parser for Uni-Sign training scripts."""
    parser = argparse.ArgumentParser('Uni-Sign Base Arguments', add_help=False)

    # --- Essential Training Parameters ---
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='Batch size per GPU (default: 8)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--output_dir', default='./output', type=str, metavar='PATH',
                        help='Path where to save checkpoints and logs (default: ./output)')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only (no training)')
    parser.add_argument('--quick_break', type=int, default=-1, metavar='N',
                        help='Break training loop after N steps for debugging (-1 to disable)')

    # --- Model Parameters ---
    parser.add_argument('--hidden_dim', default=768, type=int, metavar='DIM',
                        help='Target hidden dimension for MT5 projection (default: 768)')
    parser.add_argument('--gcn_out_dim', default=256, type=int, metavar='DIM',
                        help='Output dimension of GCN fusion modules (default: 256)')
    parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                        help='Finetune model from a checkpoint file (loads model state_dict only)')

    # --- Optimizer Parameters ---
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.999])')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9, only relevant if opt=sgd)')

    # --- LR Scheduler Parameters ---
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--warmup-epochs', type=float, default=5, metavar='N',
                        help='Epochs to warmup LR (default: 1)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='Lower LR bound for cyclic schedulers (default: 1e-6)')

    # --- Data Loading & Task ---
    parser.add_argument('--max_length', default=256, type=int, metavar='LEN',
                        help='Maximum sequence length for pose data? (default: 256)')
    parser.add_argument('--dataset', default="CSL_News", type=str,
                        choices=['CSL_News', "CSL_Daily", "PHOENIX14T", "WLASL"],
                        help='Dataset name')
    parser.add_argument('--task', default="SLT", type=str, choices=['SLT', "ISLR", "CSLR"],
                        help='Task type (SLT, ISLR, CSLR)')
    parser.add_argument('--label_smoothing', default=0.2, type=float,
                        help='Label smoothing for main CE loss (default: 0.1)')
    parser.add_argument('--pin-mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='Disable pinning CPU memory')
    parser.add_argument('--max_eval_samples', type=int, default=1000,
                    help='Maximum number of evaluation figure data samples to collect and save during a global --eval run.')

    parser.add_argument('--save_batch_name', default="testing", type=str)

    # --- Distributed Training Parameters ---
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='Url used to set up distributed training')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank for distributed training')

    # --- DeepSpeed Parameters ---
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed')

    parser.add_argument('--save_one_batch', action='store_true', help='save first batch for eval')
    parser.add_argument('--deepspeed_config', default=None, type=str, metavar='PATH',
                        help='Path to DeepSpeed config file (JSON)')
    parser.add_argument('--offload', action='store_true',
                        help='Enable ZeRO Offload techniques (CPU for optimizer/params)')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                        help='Training data type (fp16, bf16, fp32) (default: bf16)')
    parser.add_argument('--zero_stage', type=int, default=0, choices=[0, 1, 2, 3],
                        help='ZeRO optimization stage (0=disabled, 1=optimizer, 2=grads+optimizer, 3=all) (default: 0)')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, metavar='N',
                        help='Number of steps to accumulate gradients over (default: 1)')
    parser.add_argument('--gradient_clipping', default=1.0, type=float, metavar='NORM',
                        help='Gradient clipping value (norm) for DeepSpeed (default: 1.0)')

    # --- Hyperbolic Branch Parameters ---
    parser.add_argument('--use_hyperbolic', action='store_true',
                        help='Enable the hyperbolic geometry branch')
    parser.add_argument('--hyp_dim', default=256, type=int, metavar='DIM',
                        help='Dimension of hyperbolic embeddings (default: 256)')
    parser.add_argument('--init_c', default=0.5, type=float)

    parser.add_argument('--hyp_lr', default=0.001, type=float, metavar='LR',
                        help='Learning rate for hyperbolic parameters (RiemannianAdam) (default: 1e-3)')
    parser.add_argument('--hyp_stabilize', default=100, type=int, metavar='N',
                        help='Stabilization frequency for RiemannianAdam (default: 100)')
    parser.add_argument('--alpha_reg', default=0.01, type=float,
                        help='Regularization strength for alpha loss weight (default: 0.01)')


    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Regularization strength for alpha loss weight (default: 0.01)')
    parser.add_argument('--label_smoothing_hyp', default=0.2, type=float,
                        help='Label smoothing for hyperbolic contrastive loss (default: 0.1)')
    parser.add_argument('--hyp_text_emb_src', type=str, default='decoder', choices=['token', 'decoder'],
                        help="Source for text embeddings in hyperbolic loss ('token' or 'decoder') (default: 'token')")

    parser.add_argument('--hyp_text_cmp', type=str, default='token', choices=['pooled','attn', 'token'],
                        help="Source for text embeddings in hyperbolic loss ('token' or 'decoder') (default: 'token')")

    # --- Manual Gradient Clipping Parameters ---
    parser.add_argument('--manual_grad_clip', action='store_true',
                        help='Enable manual gradient clipping (Use DeepSpeed config preferably)')
    parser.add_argument('--clip_grad_norm_euclid', type=float, default=1.0,
                        help='Max norm for Euclidean gradients (if manual_grad_clip)')
    parser.add_argument('--clip_grad_norm_hyp', type=float, default=0.1,
                        help='Max norm for Hyperbolic gradients (if manual_grad_clip)')

    # --- Checkpointing ---
    parser.add_argument('--load_checkpoint_dir', default='', type=str, metavar='PATH',
                        help='Directory to load DeepSpeed checkpoint from (loads latest tag within)')

    # --- Generation Parameters ---
    parser.add_argument('--num_beams', default=4, type=int, metavar='N',
                        help='Number of beams for generation during evaluation (default: 4)')
    parser.add_argument('--max_tgt_len', default=100, type=int, metavar='LEN',
                        help='Max new tokens for generation during evaluation (default: 100)')

    # --- Logging Parameters ---
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    parser.add_argument('--rgb_support', action='store_true', help='Enable RGB features')
    parser.add_argument('--wandb_project', default='Uni-Sign-Hyperbolic-v4', type=str,
                        help='WandB project name')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Explicit WandB run name (defaults to auto-generated)')
    parser.add_argument('--run-name', dest='wandb_run_name', type=str, default=None, help=argparse.SUPPRESS) # Alias

    return parser
