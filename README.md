# AuroraNIC-MetricVisualizer

AuroraNIC-MetricVisualizer is a Python-based tool designed for processing and visualizing performance metrics for ALCF's Aurora Cassini NICs. The tool calculates raw differences between "before" and "after" metric data, organizes results by node and element counts, and generates visualizations to explore NIC performance trends.

Future enhancements will add interactive dashboards powered by Plotly for detailed and dynamic analyses.

---

## Features

- **Metrics Parsing:** Extract performance metrics from hierarchical job directories.
- **Raw Difference Analysis:** Compute "before" and "after" differences for each metric entry.
- **Heatmap Visualization:** Generate heatmaps illustrating metric differences across node counts and element counts.
- **Support for ALCF's Aurora Cassini NICs:** Optimized for analyzing Aurora NIC data.
- **Interactive Visualization (Coming Soon):** Explore NIC performance data with Plotly-based dashboards.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/AuroraNIC-MetricVisualizer.git
cd AuroraNIC-MetricVisualizer
pip install -r requirements.txt
```

---

## Usage

### 1. Directory Structure

Organize your Aurora NIC job data as follows:

```
base_directory/
└── job_dir/
    └── metrics_job_dir/
        └── node_count_num_elements/
            ├── metric_before.identifier
            └── metric_after.identifier
```

### 2. Running the Script

Set the `base_directory` in the script to point to your job data directory and run the tool:

```bash
python aurora_nic_visualizer.py
```

Heatmaps for each metric will be saved in the `figures/` directory.

---

## Heatmap Visualization

The generated heatmaps show raw differences for each metric across combinations of node counts and element counts. Each heatmap provides insights into how Aurora Cassini NIC performance varies with system configuration.

### Example Heatmap

![Heatmap Example](path/to/example_heatmap.png)

---

## Upcoming Features

- **Interactive Dashboards:** Create interactive visualizations using Plotly for:
  - Filtering by node count, element count, or metric type.
  - Exploring time-series plots for selected metrics.
  - Zooming and panning for detailed views of performance trends.

- **Dynamic CLI Configuration:** Add support for passing directory paths, filters, and output formats via command-line arguments.

- **Parallel Processing:** Optimize processing for large datasets using parallelism.

- **Exportable Results:** Save processed data as CSV or JSON files for further analysis.

---

## Contributions

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch to your fork.
4. Open a pull request with a description of your changes.

---

## License

This project is licensed under the BSD 3-Clause License. See the `LICENSE` file for details.

---

## About ALCF's Aurora Cassini NICs

Aurora, the supercomputer at the Argonne Leadership Computing Facility (ALCF), features cutting-edge Cassini network interface cards (NICs) for high-performance computing. This tool is specifically designed to help researchers and engineers analyze and improve NIC performance by providing clear and actionable visualizations of key performance metrics.

For more information about Aurora, visit [ALCF Aurora](https://www.alcf.anl.gov).

---

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue.

---

### Stay Updated!

This project is actively being developed, with major features planned for future releases. Follow the repository to stay updated on new developments!
