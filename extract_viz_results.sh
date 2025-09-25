#!/bin/bash

# Script to run viz_results.py on all replicated evaluation results
# and extract the accuracy tables

# Function to process viz_results for a directory
process_viz_results() {
    local base_dir=$1
    local lr=$2
    local dataset_type=$3
    local exp_name=$4
    local custom_dataset=${5:-""}
    
    echo "=== ${lr} - ${dataset_type} - ${exp_name} ==="
    
    for run in 1 2 3 4 5; do
        run_dir="${base_dir}/${exp_name}/run_${run}"
        
        # Find predictions.json file
        predictions_file=$(find "$run_dir" -name "predictions.json" -type f | head -1)
        
        if [ -n "$predictions_file" ]; then
            echo "Run ${run}:"
            if [ -z "$custom_dataset" ]; then
                python viz_results.py "$predictions_file" 2>/dev/null | grep -A 5 "Experiment"
            else
                python viz_results.py "$predictions_file" --custom-dataset-path "$custom_dataset" 2>/dev/null | grep -A 5 "Experiment"
            fi
            echo ""
        fi
    done
}

# Process LR=2e-5 results
echo "========================================"
echo "LEARNING RATE 2e-5 RESULTS"
echo "========================================"

# Default dataset
echo "--- DEFAULT DATASET ---"
for exp in baseline osft-chunked-decreasing-chunk{0,1,2} osft-chunked-chunk{0,1,2} osft-full sft-chunked-chunk{0,1,2} sft-full; do
    if [ -d "repeated-results-lr2e5/default/${exp}" ]; then
        process_viz_results "repeated-results-lr2e5/default" "2e-5" "default" "$exp"
    fi
done

# Single-chunk dataset
echo "--- SINGLE-CHUNK DATASET ---"
for exp in baseline-chunk{0,1,2} osft-chunked-decreasing-chunk{0,1,2} osft-chunked-chunk{0,1,2} sft-chunked-chunk{0,1,2} osft-full-chunk{0,1,2} sft-full-chunk{0,1,2}; do
    if [ -d "repeated-results-lr2e5/single-chunk/${exp}" ]; then
        # Extract chunk number from exp name
        chunk_num=$(echo "$exp" | grep -o 'chunk[0-9]' | tail -1 | grep -o '[0-9]')
        custom_dataset="/mnt/7TB-a/osilkin/EntityGraph/final_split_output/chunk_${chunk_num}/eval.json"
        process_viz_results "repeated-results-lr2e5/single-chunk" "2e-5" "single-chunk" "$exp" "$custom_dataset"
    fi
done

# Progressive dataset
echo "--- PROGRESSIVE DATASET ---"
for exp in baseline-chunk{0,1,2} osft-chunked-decreasing-chunk{0,1,2} osft-chunked-chunk{0,1,2} sft-chunked-chunk{0,1,2} osft-full-chunk{0,1,2} sft-full-chunk{0,1,2}; do
    if [ -d "repeated-results-lr2e5/progressive/${exp}" ]; then
        # Extract chunk number from exp name
        chunk_num=$(echo "$exp" | grep -o 'chunk[0-9]' | tail -1 | grep -o '[0-9]')
        custom_dataset="/mnt/7TB-a/osilkin/EntityGraph/final_split_output/chunk_${chunk_num}/eval_progressive.json"
        process_viz_results "repeated-results-lr2e5/progressive" "2e-5" "progressive" "$exp" "$custom_dataset"
    fi
done

# Process LR=5e-6 results
echo "========================================"
echo "LEARNING RATE 5e-6 RESULTS"
echo "========================================"

# Default dataset
echo "--- DEFAULT DATASET ---"
for exp in osft-full osft-chunked-chunk{0,1,2} osft-chunked-decreasing-chunk{0,1,2}; do
    if [ -d "repeated-results-lr5e6/default/${exp}" ]; then
        process_viz_results "repeated-results-lr5e6/default" "5e-6" "default" "$exp"
    fi
done

# Single-chunk dataset
echo "--- SINGLE-CHUNK DATASET ---"
for exp in osft-full-chunk{0,1,2} osft-chunked-chunk{0,1,2} osft-chunked-decreasing-chunk{0,1,2}; do
    if [ -d "repeated-results-lr5e6/single-chunk/${exp}" ]; then
        # Extract chunk number from exp name
        chunk_num=$(echo "$exp" | grep -o 'chunk[0-9]' | tail -1 | grep -o '[0-9]')
        custom_dataset="/mnt/7TB-a/osilkin/EntityGraph/final_split_output/chunk_${chunk_num}/eval.json"
        process_viz_results "repeated-results-lr5e6/single-chunk" "5e-6" "single-chunk" "$exp" "$custom_dataset"
    fi
done

# Progressive dataset
echo "--- PROGRESSIVE DATASET ---"
for exp in osft-full-chunk{0,1,2} osft-chunked-chunk{0,1,2} osft-chunked-decreasing-chunk{0,1,2}; do
    if [ -d "repeated-results-lr5e6/progressive/${exp}" ]; then
        # Extract chunk number from exp name
        chunk_num=$(echo "$exp" | grep -o 'chunk[0-9]' | tail -1 | grep -o '[0-9]')
        custom_dataset="/mnt/7TB-a/osilkin/EntityGraph/final_split_output/chunk_${chunk_num}/eval_progressive.json"
        process_viz_results "repeated-results-lr5e6/progressive" "5e-6" "progressive" "$exp" "$custom_dataset"
    fi
done
