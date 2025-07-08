# Bloom Filter & Stable Bloom Filter Project

This repository contains a Python implementation and experimental framework for **Bloom Filters (BF)** and **Stable Bloom Filters (SBF)** designed to process and deduplicate large URL datasets, such as those from Common Crawl.

## Authors

Names: M. Ropele, M. Ceola  
Date: June 3, 2025

### Background
Bloom Filters are probabilistic data structures used to efficiently test whether an element is a member of a set, with some false positive rate but no false negatives. Stable Bloom Filters extend this idea to handle streaming data, supporting approximate deletions by allowing some false negatives, useful in dynamic datasets.

### Features
- Implementations of BF and SBF with customizable parameters (`m`, `k`, `p`, etc.)
- Filtering large URL datasets to identify unique elements
- Evaluation of false positive and false negative rates
- Automatic parameter heuristics for SBF

### Storage Optimization
Saving output data in `.duckdb` format reduces storage size by approximately 50% compared to `.txt` files, due to efficient binary compression.

### Requirements
Python 3.8+  
Libraries: pandas, matplotlib, gzip, math
