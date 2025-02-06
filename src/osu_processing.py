# src/osu_processing.py

import os
import re
import numpy as np

def parse_osu_file(filepath):
    """
    Parse a single OSU Allreduce benchmark file and extract up to two lines of
    timing data.

    Each OSU Allreduce file may contain one or two data "blocks." For each block,
    a single line of numeric data is provided after a header/comment section.
    The timing metrics extracted are:

        - t_min (Min Latency in microseconds)
        - t_max (Max Latency in microseconds)
        - t_avg (Avg Latency in microseconds)
        - stddev (always set to np.nan, as OSU does not provide it)

    Returns
    -------
    list of dict
        A list of up to two dictionaries. Each dictionary has the keys:
            't_min', 't_max', 't_avg', 'stddev'.

        For example:
            [
              {'t_min': ..., 't_max': ..., 't_avg': ..., 'stddev': np.nan},
              {'t_min': ..., 't_max': ..., 't_avg': ..., 'stddev': np.nan}
            ]

        Returns an empty list if no valid data lines are found.
    """
    data_lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    block_count = 0
    i = 0
    n = len(lines)

    # Search for up to two blocks of data in the file.
    while i < n:
        line = lines[i].strip()
        # A new block typically starts with: "# OSU MPI" and contains "... Latency Test ..."
        if "OSU MPI" in line and "Latency Test" in line:
            # Move past this header line.
            i += 1
            # Skip any additional comment lines that start with '#'.
            while i < n and lines[i].strip().startswith('#'):
                i += 1

            if i < n:
                # The next non-comment line should contain numeric data:
                # Format: <Size> <Avg> <Min> <Max> <Iterations>
                parts = lines[i].split()
                i += 1
                if len(parts) >= 4:
                    try:
                        t_avg = float(parts[1])
                        t_min = float(parts[2])
                        t_max = float(parts[3])
                        data_lines.append({
                            't_min':  t_min,
                            't_max':  t_max,
                            't_avg':  t_avg,
                            'stddev': np.nan
                        })
                        block_count += 1
                    except ValueError:
                        pass  # Skip malformed lines

            # OSU typically has up to two blocks; stop if we've seen both.
            if block_count == 2:
                break
        else:
            i += 1

    return data_lines


def process_osu_benchmarks(base_dir):
    """
    Parse OSU Allreduce benchmark data from multiple job directories.

    For each job directory in base_dir, this function searches an "out_{job_dir}"
    subdirectory for files named in the pattern:

        osu_allreduce_{job_name}_{node_count}_{ranks}_{rpn}_{size}_out.txt

    The 'node_count' is inferred from the matching filename, and 'size' indicates
    the element count. Each matching file is parsed for timing data via
    parse_osu_file().

    Returns
    -------
    dict
        A dictionary of the form:
            {
              node_count: {
                'element_counts': [size1, size2, ...],
                'data': np.array of shape (num_elements, 3, 4)
              },
              ...
            }

        where:

        - 'element_counts' is a sorted list of the sizes found for that node_count
        - 'data' is a NumPy array of shape (num_elements, 3, 4)

          The second dimension (3) corresponds to up to three lines of timing data.
          For OSU, only the first two entries contain parsed metrics (if available),
          and the third entry is filled with np.nan. The third dimension (4)
          corresponds to [t_min, t_max, t_avg, stddev].

    Notes
    -----
    This structure mirrors the output of process_oneccl_benchmarks() so that it
    can be passed directly to prepare_benchmark_array(...) without any changes.
    """
    bench_data = {}

    for job_dir in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_dir)
        out_dir = os.path.join(job_path, f"out_{job_dir}")
        if not os.path.isdir(out_dir):
            continue

        # Initialize values used per directory (node_count parsed from matching filenames)
        node_count = None
        file_map = {}

        # The naming pattern for OSU Allreduce is:
        #   osu_allreduce_{job_name}_{node_count}_{ranks}_{rpn}_{size}_out.txt
        pattern = r"osu_allreduce_([^_]+)_(\d+)_(\d+)_(\d+)_(\d+)_out\.txt"

        # Gather all files that match our pattern
        for fname in os.listdir(out_dir):
            match = re.match(pattern, fname)
            if not match:
                continue

            # Example filename:
            #   osu_allreduce_1264191.aurora_2048_24576_12_268435456_out.txt
            # Corresponding groups:
            #   (job_name, node_count, ranks, rpn, size)
            # job_name = match.group(1)     # e.g. 126419.aurora
            ncount = int(match.group(2))    # e.g. 2048
            # ranks = int(match.group(3))   # e.g. 24576
            # rpn = int(match.group(4))     # e.g. 12
            size_val = int(match.group(5))  # e.g. 268435456

            if node_count is None:
                node_count = ncount
            if node_count != ncount:
                # Skip if we encounter a different node_count in the same directory
                continue

            file_map[size_val] = os.path.join(out_dir, fname)

        if node_count is None or not file_map:
            # No matching OSU files found in this directory
            continue

        # Sort sizes just like we do with element counts
        sorted_sizes = sorted(file_map.keys())

        # Prepare data array: shape = (num_sizes, 3, 4)
        # Where 3 is the up to 3 lines of timing data, and 4 is [t_min, t_max, t_avg, stddev].
        data_array = np.zeros((len(sorted_sizes), 3, 4), dtype=np.float64)

        for i, size_val in enumerate(sorted_sizes):
            filepath = file_map[size_val]
            parsed_data = parse_osu_file(filepath)

            # parsed_data can have 0, 1, or 2 entries (each a dict).
            # We only fill up to two rows; the third row is filled with NaN.
            if len(parsed_data) == 0:
                data_array[i, :, :] = np.nan
            else:
                # Fill entries for each block found
                for j in range(3):
                    if j < len(parsed_data):
                        d = parsed_data[j]
                        data_array[i, j, 0] = d['t_min']
                        data_array[i, j, 1] = d['t_max']
                        data_array[i, j, 2] = d['t_avg']
                        data_array[i, j, 3] = d['stddev']
                    else:
                        data_array[i, j, :] = np.nan

        # Store the data for this node_count in the same structure as oneCCL
        bench_data[node_count] = {
            'element_counts': sorted_sizes,
            'data': data_array
        }

    return bench_data
