#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Assign the directory path to a variable
DIR=$1

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist."
    exit 1
fi

# Iterate over .gz files and unzip them
for file in "$DIR"/*.gz; do
    if [ -f "$file" ]; then
        echo "Unzipping $file..."
        gunzip "$file"
    fi
done

echo "Unzipping complete."
