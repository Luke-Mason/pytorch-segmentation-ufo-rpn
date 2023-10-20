#!/bin/bash

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

# Extract the <number> from the argument
arg=$1
number=$(echo $arg | sed 's/\([0-9]\)/\1/')

# Check if the argument matches the expected format
if [[ ! $arg =~ ^[0-9]+$ ]]; then
    echo "Invalid argument format. Expected a digit."
    exit 1
fi

# Loop from 0 to 9 (inclusive) and call the Python script
for i in {0..9}; do
    config="dstl_ex${number}.json"
    echo "Calling python3 ../dstl_train.py --config $config --cl $i"
    python3 ../dstl_train.py --config "$config" --cl "$i"
done
