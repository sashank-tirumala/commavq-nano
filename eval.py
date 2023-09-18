import pdb

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from model import GPT, GPTConfig  # Import your custom model here

## constants
BOS_TOKEN = 1024
TOKENS_PER_FRAME = 129
BS = 10
CONTEXT_SIZE_FRAMES = 20
N_FRAMES = 1200
N = N_FRAMES - 20


def main():
    # System configuration
    device = "cuda"
    compile = True
    seed_offset = 0
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn

    # Load pre-trained GPT model
    ckpt_path = "/home/sudatha/rob/nanoGPT/out-commavq_2/ckpt.pt"
    model = get_eval_model(ckpt_path=ckpt_path, device=device)

    # Load evaluation data files and indices
    files, indices = get_eval_files_indices()

    total_losses = []
    losses, sizes = [], []
    pbar = tqdm(files)
    for f in pbar:
        tokens = np.load(f)
        tokens = preprocess_tokens(tokens)

        for ii in indices:
            loss, size = compute_loss(model, tokens, ii)
            losses.append(loss)
            sizes.append(size)

    total_loss = np.sum(losses) / np.sum(sizes)
    total_losses.append(total_loss)
    print(f"Total loss: {np.mean(total_losses)}")


def get_eval_model(ckpt_path, device):
    """
    Initialize and load a pre-trained GPT model.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (str): Device to use for the model ('cpu', 'cuda', 'cuda:0', etc.).

    Returns:
        GPT: Loaded GPT model.
    """
    # Model configuration and loading
    init_config = GPTConfig()
    model_args = initialize_model_args(init_config)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    model_args = update_model_args(model_args, checkpoint_model_args)
    # Create and load the model
    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig)
    state_dict = checkpoint["model"]
    state_dict = fix_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def initialize_model_args(init_config):
    """
    Initialize the model configuration arguments by updating them from a checkpoint.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        init_config (GPTConfig): Initial GPT configuration.

    Returns:
        dict: Updated model configuration arguments.
    """

    model_args = dict(
        n_layer=init_config.n_layer,
        n_head=init_config.n_head,
        n_embd=init_config.n_embd,
        block_size=init_config.block_size,
        bias=init_config.bias,
        vocab_size=None,
        dropout=init_config.dropout,
    )
    return model_args


def update_model_args(model_args, checkpoint_model_args):
    """
    Update the model configuration arguments by updating them from a checkpoint model arguments

    Args:

        init_config (GPTConfig): Initial GPT configuration.
        checkpoint_config (GPTConfig): Checkpoint GPT configuration

    Returns:
        dict: Updated model configuration arguments.
    """
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    return model_args


def fix_state_dict_keys(state_dict):
    """
    Fix the keys of the state dictionary.

    Args:
        state_dict (dict): State dictionary of the model.

    Returns:
        dict: State dictionary with fixed keys.
    """

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


def get_eval_files_indices(data_path="commaai/commavq"):
    """
    Load evaluation dataset and generate indices for slicing data.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        list: List of file paths.
        list: List of data indices for slicing.
    """
    ds = load_dataset(data_path, num_proc=40)
    files = ds["40"]["path"]  # 40 is the validation set

    # Create data slicing
    indices = np.arange(0, N * TOKENS_PER_FRAME)
    indices = np.array(np.split(indices, N // CONTEXT_SIZE_FRAMES))
    # Batch them
    indices = [indices[i : i + BS] for i in range(0, len(indices), BS)]

    return files, indices


def preprocess_tokens(tokens):
    """
    Preprocess tokens for model input.

    Args:
        tokens (numpy.ndarray): Input tokens.

    Returns:
        torch.Tensor: Processed tokens as a PyTorch tensor.
    """
    tokens = tokens.reshape(
        N_FRAMES, TOKENS_PER_FRAME - 1
    )  # TOKENS_PER_FRAME includes the BOS token
    tokens = np.c_[np.ones(len(tokens), dtype=np.int64) * BOS_TOKEN, tokens]
    tokens = tokens.reshape(-1)
    tokens = torch.from_numpy(tokens).long().cuda()
    return tokens


def compute_loss(model, tokens, indices):
    """
    Compute the loss for a batch of tokens.

    Args:
        model (GPT): GPT model.
        tokens (torch.Tensor): Input tokens.
        indices (numpy.ndarray): Data indices for slicing.

    Returns:
        float: Loss for the batch.
        int: Batch size.
    """
    with torch.no_grad():
        x = tokens[indices.ravel()]
        x = x.reshape(indices.shape[0], indices.shape[1])
        pred, _ = model(x, comma_vq=True)
        y = tokens[indices.ravel() + 1]
        y = y.reshape(indices.shape[0], indices.shape[1])
        loss = (
            F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
            .detach()
            .cpu()
            .numpy()
            * indices.shape[0]
        )
        return loss, indices.shape[0]


if __name__ == "__main__":
    main()
