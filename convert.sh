#!/bin/bash

# Get script location 
DATA_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/data"

# Create merged data 
echo -n "" > "$DATA_DIR/merged.dna"

echo "Producing merged file: merged.dna"
for f in "$DATA_DIR"/*.dnap
do
    exec 3< "$f" || { echo "Cannot open positive file ${pos_filename}" >&2; exit 1; }
    exec 4< "${f/%dnap/dnan}" || { echo "Cannot open positive file ${pos_filename}" >&2; exit 1; }
    while read -r pos <&3 && read -r neg <&4 >&/dev/null
    do
        echo -e "1\t${pos}\n0\t${neg}" >> "${DATA_DIR}/merged.dna"
    done
done

# Shuffle merged data
echo "Producing shuffled file: shuffled.dna"
shuf "$DATA_DIR/merged.dna" > "$DATA_DIR/shuffled.dna"

merge_size=$(cat "$DATA_DIR"/shuffled.dna | wc -l)
train_count=$((merge_size*75/100))
test_count=$((merge_size*20/100))
valid_count=$((merge_size*5/100))

# Create training file (random 75%)
echo "Producing training file: train.dna"
head -n "$train_count" "$DATA_DIR/shuffled.dna" > "$DATA_DIR/train.dna"
head -n 80 "$DATA_DIR/train.dna" > "$DATA_DIR/train-light.dna"

# Create test file (random 20%)
echo "Producing testing file: test.dna"
sed -n $((train_count+1)),$((train_count+test_count+1))p "$DATA_DIR/shuffled.dna" > "$DATA_DIR/test.dna"
echo "Producing lightweight testing file: test-light.dna"
head -n 20 "$DATA_DIR/test.dna" > "$DATA_DIR/test-light.dna"

# Create validation file (random 5%)
echo "Producing validation file: valid.dna"
tail -n "$valid_count" "$DATA_DIR/shuffled.dna" > "$DATA_DIR/valid.dna"
