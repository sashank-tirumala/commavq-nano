# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-commavq"
eval_interval = 500  # 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True  # override via command line if you like
wandb_project = "commavq"
wandb_run_name = "mini-gpt"

dataset = "commavq"
gradient_accumulation_steps = 1
batch_size = 12  ## based on number of parameters and compute size
block_size = 129 * 20  # 129 tokens per frame times the number of context frames.

# baby GPT model :)
n_layer = 8  # 8
n_head = 8  # 8
n_embd = 512  # 512 #1024#384
dropout = 0.2

learning_rate = 1e-4  # with baby networks can afford to go a bit higher
max_iters = 500000  # 2000000
lr_decay_iters = 500000  # make equal to max_iters usually
min_lr = 1e-5  # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small, uncomment if the tokens/iteration is small.

warmup_iters = 2000  # not super necessary potentially
## vocab size = 1048
