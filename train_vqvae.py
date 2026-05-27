import os
import torch
import torch.distributed as dist
from trainer import VQVAETrainer
from dataset import CombinedData
from utils import get_se_args

# Resolve OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Parse arguments
args = get_se_args()

# torchrun automatically sets CUDA_VISIBLE_DEVICES; manual setting is unnecessary
# Create project directory if it doesn't exist
if not os.path.exists(args.save_dir):
    # Fixed code
    os.makedirs(args.save_dir, exist_ok=True)

def sync_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def run(args):
    # Get DDP environment variables (torchrun sets these automatically)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    
    # Check if DDP mode is enabled
    multi_gpu = world_size > 1
    
    # Print info only on rank 0
    if rank == 0:
        print(f"{'='*60}")
        if multi_gpu:
            print(f"DDP training mode")
            print(f"Total number of processes: {world_size}")
        else:
            print(f"Single-GPU training mode")
        print(f"Batch size per GPU: {args.batch_size}")
        if multi_gpu:
            print(f"Total batch size: {args.batch_size * world_size}")
        print(f"{'='*60}")
    
    # Initialize datasets
    train_dataset = CombinedData(args.data_list, args.surface_list, args.edge_list, 
                                  validate=False, aug=True, use_type_flag=args.use_type_flag,
                                  surface_mmap=args.surface_mmap,
                                  edge_mmap=args.edge_mmap)
    
    # Initialize trainer first so all ranks join the DDP process group together.
    vae = VQVAETrainer(args, train_dataset, None, multi_gpu=multi_gpu)
    
    # Load validation data only after DDP initialization; other ranks wait at the barrier.
    if rank == 0:
        val_dataset = CombinedData(args.data_list, args.surface_list, args.edge_list, 
                                    validate=True, aug=False, use_type_flag=args.use_type_flag,
                                    val_surface_cache=args.val_surface_cache,
                                    val_edge_cache=args.val_edge_cache,
                                    val_surface_mmap=args.val_surface_mmap,
                                    val_edge_mmap=args.val_edge_mmap,
                                    max_items=args.val_max_items)
        vae.set_val_dataset(val_dataset)
    sync_if_needed()
    
    # After trainer initialization, DDP is ready; safe to use dist functions
    if rank == 0:
        print(f'Starting training from epoch: {vae.epoch}')
        print(f'Target epoch: {args.train_epoch}')
        print(f"{'='*60}")
    
    # Training loop
    while vae.epoch <= args.train_epoch:
        # Save current epoch number (before incrementing)
        current_epoch = vae.epoch
        
        # Train one epoch (internally increments vae.epoch and handles synchronization at the end)
        vae.train_one_epoch()
        
        # Validation is rank 0 only; all ranks wait afterwards before the next epoch.
        if current_epoch % args.test_epoch == 0:
            if rank == 0:
                vae.test_val()
            sync_if_needed()
        
        # Saving is rank 0 only; all ranks wait afterwards to keep DDP steps aligned.
        if current_epoch % args.save_epoch == 0:
            if rank == 0:
                vae.save_model(save_epoch=current_epoch)
            sync_if_needed()
    
    # Save final model (if not already saved)
    if rank == 0:
        final_epoch = vae.epoch - 1
        # Check if the last epoch has already been saved
        if final_epoch % args.save_epoch != 0:
            vae.save_model(save_epoch=final_epoch)
        vae.close_writer()
        print(f"{'='*60}")
        print(f'Training completed! Final epoch: {final_epoch}')
        print(f"{'='*60}")
           

if __name__ == "__main__":
    run(args)