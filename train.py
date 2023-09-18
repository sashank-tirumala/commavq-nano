import math
import os
import pdb
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig

master_process = True
seed_offset = 0
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


# Define constants and configuration
def initialize_config():
    """
    Initialize configuration parameters as global variables.
    """
    config = {
        "out_dir": "out",
        "eval_interval": 2000,
        "log_interval": 1,
        "eval_iters": 200,
        "eval_only": False,
        "always_save_checkpoint": True,
        "init_from": "scratch",
        "wandb_log": False,
        "wandb_project": "commavq",
        "wandb_run_name": "nanogpt",
        "dataset": "commavq",
        "gradient_accumulation_steps": 5 * 8,
        "batch_size": 12,
        "block_size": 1024,
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "dropout": 0.2,
        "bias": False,
        "learning_rate": 6e-4,
        "max_iters": 600000,
        "weight_decay": 1e-1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "decay_lr": True,
        "warmup_iters": 2000,
        "lr_decay_iters": 600000,
        "min_lr": 6e-5,
        "device": "cuda",
        "dtype": "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16",
        "compile": True,
    }

    for key, value in config.items():
        globals()[key] = value


# Load and preprocess data
def load_data(data_dir):
    """
    Load and preprocess training and validation data.

    Args:
        data_dir (str): Directory containing data files.

    Returns:
        np.memmap: Training data.
        np.memmap: Validation data.
    """
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


def get_batch(split):
    """
    Get a batch of training or validation data.

    Args:
        split (str): Either 'train' or 'val' to specify the dataset split.

    Returns:
        torch.Tensor: Input data tensor.
        torch.Tensor: Target data tensor.
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Model initialization and training
def initialize_model(config, data_dir):
    """
    Initialize the model, optimizer, scaler, and other components for training.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        data_dir (str): Directory path to the training data.

    Returns:
        torch.nn.Module: Initialized model.
        torch.optim.Optimizer: Initialized optimizer.
        torch.cuda.amp.GradScaler: Initialized GradScaler.
        int: Current iteration number.
        float: Best validation loss.
    """
    model_args = dict(
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        block_size=config["block_size"],
        bias=config["bias"],
        vocab_size=None,
        dropout=config["dropout"],
    )

    # Initialize model based on 'init_from' option
    if config["init_from"] == "scratch":
        # Initialize a new model from scratch
        print("Initializing a new model from scratch")
        meta_path = os.path.join(data_dir, "meta.pkl")
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        if meta_vocab_size is None:
            print(
                "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
            )
        model_args["vocab_size"] = (
            meta_vocab_size if meta_vocab_size is not None else 1048
        )
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = 1e9
    elif config["init_from"] == "resume":
        print(f"Resuming training from {config['out_dir']}")
        # Resume training from a checkpoint
        ckpt_path = os.path.join(config["out_dir"], "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config["device"])
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    # Crop down the model block size if necessary
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        model_args["block_size"] = config["block_size"]

    model.to(config["device"])

    # Initialize a GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(config["dtype"] == "float16"))

    # Initialize the optimizer
    optimizer = model.configure_optimizers(
        config["weight_decay"],
        config["learning_rate"],
        (config["beta1"], config["beta2"]),
        config["device"],
    )
    if config["init_from"] == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None

    # Compile the model if required
    if config["compile"]:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    return model, optimizer, scaler, iter_num, best_val_loss


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss on the training and validation datasets.

    Returns:
        dict: A dictionary containing the estimated losses for 'train' and 'val' splits.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, comma_vq=True)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    """
    Calculate the learning rate for a given iteration.

    Args:
        it (int): Current iteration number.

    Returns:
        float: The learning rate for the current iteration.
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# Training loop
def train_model(model, optimizer, scaler, config, iter_num, best_val_loss):
    """
    Train the model using the specified configuration.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scaler (torch.cuda.amp.GradScaler): GradScaler for mixed-precision training.
        config (dict): Configuration parameters for training.
        iter_num (int): Current iteration number.
        best_val_loss (float): Best validation loss achieved so far.
    """

    if config["wandb_log"] and master_process:
        import wandb

        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config=config,
        )

    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if config["decay_lr"] else config["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config["eval_interval"] == 0 and master_process:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if config["wandb_log"]:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }
                )
            if losses["val"] < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": config["model_args"],
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config["out_dir"], "ckpt.pt"))
        if iter_num == 0 and config["eval_only"]:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(config["gradient_accumulation_steps"]):
            with ctx:
                logits, loss = model(X, Y, comma_vq=False)
                loss = (
                    loss / config["gradient_accumulation_steps"]
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if config["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config["log_interval"] == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * config["gradient_accumulation_steps"]
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(
                    config["batch_size"] * config["gradient_accumulation_steps"], dt
                )  ##
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config["max_iters"]:
            break


if __name__ == "__main__":
    initialize_config()
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    exec(open("configurator.py").read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging

    tokens_per_iter = (
        config["gradient_accumulation_steps"]
        * config["batch_size"]
        * config["block_size"]
    )
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(config["out_dir"], exist_ok=True)

    data_dir = os.path.join("data", config["dataset"])
    train_data, val_data = load_data(data_dir)
    model, optimizer, scaler, iter_num, best_val_loss = initialize_model(
        config, data_dir
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    ctx = (
        nullcontext()
        if config["device"] == "cpu"
        else torch.amp.autocast(device_type=config["device"], dtype=ptdtype)
    )

    train_model(
        model, optimizer, scaler, config, iter_num=iter_num, best_val_loss=best_val_loss
    )
