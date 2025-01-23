# src/oneccl_processing.py

import os
import re
import numpy as np

def parse_oneccl_file(filepath):
    """
    Parse a single oneCCL benchmark file and extract three lines of timing data.

    Returns:
        A list of up to three dictionaries, each with the keys:
            't_min', 't_max', 't_avg', 'stddev'.
        Returns an empty list if parsing fails or if data is incomplete.
    """
    data_lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the performance table header
    header_index = None
    for i, line in enumerate(lines):
        if '#bytes' in line and 't_min[usec]' in line:
            header_index = i + 1
            break
    if header_index is None:
        return data_lines

    # Extract up to three lines of data following the header
    for i in range(3):
        idx = header_index + i
        if idx < len(lines):
            row = lines[idx].strip()
            if not row or row.startswith('#'):
                continue
            parts = row.split()
            if len(parts) < 7:
                continue
            try:
                t_min = float(parts[3])
                t_max = float(parts[4])
                t_avg = float(parts[5])
                stddev = float(parts[6])
                data_lines.append({
                    't_min': t_min,
                    't_max': t_max,
                    't_avg': t_avg,
                    'stddev': stddev
                })
            except ValueError:
                pass

    return data_lines


def process_oneccl_benchmarks(base_dir):
    """
    Parse oneCCL benchmark data from multiple job directories.

    For each job directory, searches an "out_{job_dir}" subdirectory for
    files named in the pattern:
        oneccl_allreduce_{job_name}_{node_count}_{ranks}_{rpn}_{elem}_sycl_ccl_gpu_out_w1.txt

    The 'node_count' is inferred from the matching filename, and 'elem' indicates
    the element count. Each file is parsed for timing data via parse_oneccl_file().

    Returns:
        A dictionary of the form:
            {
              node_count: {
                'element_counts': [...],
                'data': np.array of shape (num_elements, 3, 4)
              },
              ...
            }
        where the 3 dimension corresponds to the three lines of timing data,
        and 4 corresponds to (t_min, t_max, t_avg, stddev).
    """
    bench_data = {}

    for job_dir in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_dir)
        out_dir = os.path.join(job_path, f"out_{job_dir}")
        if not os.path.isdir(out_dir):
            continue

        # Parse node_count from matching filenames
        node_count = None
        file_map = {}

        for fname in os.listdir(out_dir):
            if not fname.startswith("oneccl_allreduce_") or not fname.endswith("_sycl_ccl_gpu_out_w1.txt"):
                continue

            # Example filename:
            #   oneccl_allreduce_10281099.amn-0001_64_768_12_1024_sycl_ccl_gpu_out_w1.txt
            # Pattern: oneccl_allreduce_{job_name}_{node_count}_{ranks}_{rpn}_{elem}_sycl_ccl_gpu_out_w1.txt
            pattern = r"oneccl_allreduce_([^_]+)_(\d+)_(\d+)_(\d+)_(\d+)_sycl_ccl_gpu_out_w1\.txt"
            match = re.match(pattern, fname)
            if not match:
                continue

            # job_name = match.group(1)     # e.g. 10281099.amn-0001
            ncount = int(match.group(2))    # e.g. 64
            # ranks = int(match.group(3))   # e.g. 768
            # rpn = int(match.group(4))     # e.g. 12
            elem_count = int(match.group(5))  # e.g. 1024

            if node_count is None:
                node_count = ncount
            if node_count != ncount:
                # Skip files that have a mismatched node_count
                continue

            file_map[elem_count] = os.path.join(out_dir, fname)

        if node_count is None:
            continue

        sorted_elems = sorted(file_map.keys())
        if not sorted_elems:
            continue

        # Prepare an array to store 3 lines x 4 stats
        data_array = np.zeros((len(sorted_elems), 3, 4), dtype=np.float64)

        for i, elem_count in enumerate(sorted_elems):
            fpath = file_map[elem_count]
            lines_data = parse_oneccl_file(fpath)
            if len(lines_data) == 3:
                for j, d in enumerate(lines_data):
                    data_array[i, j, 0] = d['t_min']
                    data_array[i, j, 1] = d['t_max']
                    data_array[i, j, 2] = d['t_avg']
                    data_array[i, j, 3] = d['stddev']
            else:
                # Fill with NaN if data is incomplete
                data_array[i, :, :] = np.nan

        bench_data[node_count] = {
            'element_counts': sorted_elems,
            'data': data_array
        }

    return bench_data

def prepare_benchmark_array(bench_results):
    """
    Reorganize the benchmark data into a consolidated NumPy array.

    Parameters
    ----------
    bench_results : dict
        A dictionary of the form returned by process_oneccl_benchmarks, i.e.:
        {
          node_count: {
            'element_counts': [e1, e2, ...],
            'data': np.array of shape (num_elems, 3, 4)
                    where:
                      - num_elems is len(element_counts)
                      - the second dimension (3) corresponds to 3 iterations
                      - the third dimension (4) corresponds to [t_min, t_max, t_avg, stddev]
          },
          ...
        }

    Returns
    -------
    bench_array : np.ndarray
        A 4D array of shape:
            (num_node_counts, num_element_counts, 3, 4)
        where:
          - The first axis is over sorted node_counts
          - The second axis is over sorted unique element_counts
          - The third axis is the iteration index (0..2)
          - The fourth axis is the statistic index:
                0 -> t_min
                1 -> t_max
                2 -> t_avg
                3 -> stddev
          Any missing (node_count, element_count) pair will be filled with np.nan.

    node_counts : list of int
        Sorted list of node counts extracted from bench_results.

    element_counts : list of int
        Sorted list of all unique element counts combined across all node counts.

    Notes
    -----
    - This function merges all node counts and element counts from bench_results
      into a single consistent 4D array for simpler heatmap or line-plot usage.
    - For a node_count that does not have a particular element_count, the
      corresponding array slice is filled with np.nan.
    """
    # 1. Gather all node_counts and sort them.
    all_node_counts = sorted(bench_results.keys())

    # 2. Gather all element_counts from all node_counts and build a global sorted list.
    all_element_sets = []
    for nc in all_node_counts:
        all_element_sets.append(bench_results[nc]['element_counts'])
    # Flatten and deduplicate
    unique_elements = sorted(set().union(*all_element_sets))

    # 3. Prepare the 4D array:
    #    shape = (num_node_counts, num_element_counts, 3, 4)
    #    Fill with np.nan by default.
    num_nodes = len(all_node_counts)
    num_elems = len(unique_elements)
    bench_array = np.full((num_nodes, num_elems, 3, 4), np.nan, dtype=np.float64)

    # 4. Build a small helper index for quick lookup:
    #    For each node_count, 'element_counts' is already sorted,
    #    so create a dict: {elem_count -> row index in the data array}.
    #    The map allows data to be easily copied into the correct location in bench_array.
    for i, node_count in enumerate(all_node_counts):
        elem_list = bench_results[node_count]['element_counts']
        data_3d = bench_results[node_count]['data']  # shape = (nLocal, 3, 4)
        # Build a map from elem -> local row index.
        local_index_map = {elem_val: idx for idx, elem_val in enumerate(elem_list)}

        # Fill bench_array for current node_count
        for j, global_elem in enumerate(unique_elements):
            if global_elem in local_index_map:
                local_idx = local_index_map[global_elem]
                # data_3d[local_idx] is shape (3,4) for the 3 iterations x 4 stats
                bench_array[i, j, :, :] = data_3d[local_idx, :, :]

    return bench_array, all_node_counts, unique_elements
