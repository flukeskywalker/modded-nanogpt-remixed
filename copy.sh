#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 5 ] || [ "$1" != "ssh" ] || [ "$3" != "-p" ]; then
    echo "Usage: $0 ssh user@host -p port"
    exit 1
fi

REMOTE="$2"
PORT="$4"

REMOTE_FILE="$5"
LOCAL_DIR="./logs"

# Ensure local logs directory exists
mkdir -p "$LOCAL_DIR"

# Copy the file from the remote machine to the local logs directory
scp -P "$PORT" "$REMOTE:$REMOTE_FILE" "$LOCAL_DIR/"
