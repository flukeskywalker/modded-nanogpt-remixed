#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 4 ] || [ "$1" != "ssh" ] || [ "$3" != "-p" ]; then
    echo "Usage: $0 ssh user@host -p port"
    exit 1
fi

REMOTE="$2"
PORT="$4"


# Install required packages on the remote
ssh -p "$PORT" "$REMOTE" "sudo apt update && sudo apt install -y rsync curl tmux htop vim tree"

# Sync the current directory contents with the remote home directory
rsync -avz -e "ssh -p $PORT" --exclude '.*' --exclude 'log' --exclude 'img' ./ "$REMOTE":~

# Start a tmux session on the remote and execute commands inside it
ssh -p "$PORT" "$REMOTE" "tmux new-session -d -s setup"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'curl -LsSf https://astral.sh/uv/install.sh | sh' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'source \$HOME/.local/bin/env' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'uv venv .venv --python=3.12' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'source .venv/bin/activate' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'uv pip install numpy tqdm huggingface-hub' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'python data/cached_fineweb10B.py 60' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'unset LD_LIBRARY_PATH' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'torchrun --standalone --nproc_per_node=8 train_gpt_M.py' C-m"
ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'torchrun --standalone --nproc_per_node=8 train_gpt_A.py' C-m"

# ssh -p "$PORT" "$REMOTE" "tmux send-keys -t setup 'torchrun --standalone --nproc_per_node=8 train_gpt2.py \
#     --input_bin "data/fineweb10B/fineweb_train_*.bin" \
#     --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
#     --output_dir pylog124M \
#     --model d12 \
#     --batch_size 64 \
#     --sequence_length 1024 \
#     --val_loss_every 128 \
#     --num_iterations 9536 \
#     --weight_decay 0.1 \
#     --learning_rate 0.0018 \
#     --warmup_iters 256 \
#     --warmdown_iters 2048' C-m"

# Attach to the remote tmux session
# ssh -p "$PORT" "$REMOTE" "tmux attach -t setup"