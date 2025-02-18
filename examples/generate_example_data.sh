#!/bin/bash

# Base directory
BASE_DIR="./examples/jobs/oneccl"

# Define number of steps for nodes and elements
NUM_NODES=3            # Number of times to multiply by 2 for nodes
NUM_ELEMENT_COUNTS=3   # Number of times to multiply by 10 for elements

# Define number of metrics and interfaces
NUM_METRICS=5
NUM_INTERFACES=8

# Define ranks per node
RPN=12

# Generate metric_names.txt file
METRIC_NAMES_FILE="metric_names.txt"
> "$METRIC_NAMES_FILE"  # Clear existing file if it exists

for ((M = 0; M < NUM_METRICS; M++)); do
    echo "metric-${M}" >> "$METRIC_NAMES_FILE"
done

# Iterate over node counts (powers of 2)
NODE=1
for ((NODE_INDEX = 0; NODE_INDEX < NUM_NODES; NODE_INDEX++)); do
    # Define job name
    JOB_NAME="job-name-${NODE}"
    
    # Create metric output directory
    METRIC_DIR="${BASE_DIR}/${JOB_NAME}/metrics_${JOB_NAME}"
    mkdir -p "${METRIC_DIR}"
    
    # Create benchmark output directory
    BENCH_DIR="${BASE_DIR}/${JOB_NAME}/out_${JOB_NAME}"
    mkdir -p "$BENCH_DIR"
    
    # Iterate over element counts (powers of 10)
    ELEM=10
    for ((ELEM_INDEX = 0; ELEM_INDEX < NUM_ELEMENT_COUNTS; ELEM_INDEX++)); do

        # Create subdirectory
        SUBDIR="${METRIC_DIR}/${NODE}_${ELEM}"
        mkdir -p "$SUBDIR"

        # Generate metric files based on node count
        for ((NODE_ID = 0; NODE_ID < NODE; NODE_ID++)); do
            AFTER_FILE="${SUBDIR}/metric_after.node${NODE_ID}"
            BEFORE_FILE="${SUBDIR}/metric_before.node${NODE_ID}"

            # Clear existing files
            > "$AFTER_FILE"
            > "$BEFORE_FILE"

            # Populate files
            for ((M = 0; M < NUM_METRICS; M++)); do
                for ((I = 0; I < NUM_INTERFACES; I++)); do
                    # metric_before content
                    echo "0@${NODE}.${I}" >> "$BEFORE_FILE"
                    
                    # Compute flattened 4D index
                    INDEX=$(( (((NODE_INDEX * NUM_ELEMENT_COUNTS + ELEM_INDEX) * NUM_METRICS + M) * NUM_INTERFACES) + I ))
                    
                    # metric_after content
                    echo "${INDEX}@${NODE}.${I}" >> "$AFTER_FILE"
                done
            done
        done

        # Generate synthetic oneCCL benchmark output files
        for ((NODE_ID = 0; NODE_ID < NODE; NODE_ID++)); do
            RANKS=$((NODE * RPN))  # Simulated ranks count
            
            OUT_FILE="${BENCH_DIR}/oneccl_allreduce_${JOB_NAME}_${NODE}_${RANKS}_${RPN}_${ELEM}_sycl_ccl_gpu_out_w1.txt"

            echo "# Example oneCCL benchmark output" > "$OUT_FILE"
            echo "#bytes   N/A   N/A   t_min[usec]  t_max[usec]  t_avg[usec]  stddev" >> "$OUT_FILE"

            for ((i = 0; i < 3; i++)); do
                T_MIN=$(echo "scale=3; 10 + ($RANDOM % 40) + ($RANDOM/32767)" | bc)
                T_MAX=$(echo "scale=3; 50 + ($RANDOM % 50) + ($RANDOM/32767)" | bc)
                T_AVG=$(echo "scale=3; 20 + ($RANDOM % 60) + ($RANDOM/32767)" | bc)
                STDDEV=$(echo "scale=3; 1 + ($RANDOM % 4) + ($RANDOM/32767)" | bc)

                printf "1024   0   0   %.3f   %.3f   %.3f   %.3f\n" "$T_MIN" "$T_MAX" "$T_AVG" "$STDDEV" >> "$OUT_FILE"
            done
        done

        # Multiply elements count by 10 for the next iteration
        ELEM=$((ELEM * 10))

    done

    # Multiply node count by 2 for the next iteration
    NODE=$((NODE * 2))

done

echo "Directory structure and files created successfully."
