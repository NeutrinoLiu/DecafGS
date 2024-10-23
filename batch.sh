#!/bin/bash

# Function to clean up and exit
cleanup_and_exit() {
    echo "Script interrupted. Cleaning up and exiting..."
    exit 1
}

# Set up trap to catch interruption signals
trap cleanup_and_exit SIGINT SIGTERM

# Loop through idx values from 1 to 100
for idx in 42 2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 101 103 107 109 113 127 131 137 139 149 151 157 163 167 173 179 181 191 193 197 199 211 223 227 229 233 239 241 251 257 263 269 271 277 281 283 293; do
    echo "Running iteration $idx"
    python train.py train.random_seed=$idx \
        train.log_path="./res/$idx.json" \
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred in iteration $idx. Exiting..."
        exit 1
    fi
done

echo "All iterations completed successfully."