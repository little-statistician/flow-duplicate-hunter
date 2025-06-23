import random
from collections import Counter
import hashlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import gzip
import requests     # Used for HTTP requests
import json
import time
import itertools

os.chdir(r'...\path\dir') 

############################# BLOOM FILTER

class BloomFilter:
    def __init__(self, m, k):
        self.m = m       # Size of the bit array
        self.k = k       # Number of hash functions
        self.bf = [0] * m  # Initialize bit array with zeros

    def _compute_hash(self, item):
        """Generate k independent hashes using double hashing"""
        hash_values = []

        # Compute base hashes
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)

        # Derive positions in the bit array
        for i in range(self.k):
            hash_values.append((h1 + i * h2) % self.m)

        return hash_values

    def add(self, item):
        """Insert the element into the filter by setting the computed bits to 1"""
        hash_values = self._compute_hash(item)

        i = 0
        while i < self.k:  # len(hash_values) == k
            self.bf[hash_values[i]] = 1
            i += 1

    def check(self, item):
        """
        Check for potential presence of the element.
        Returns True if all corresponding bits are set (possible presence),
        False if at least one bit is not set (definite absence).
        """
        hash_values = self._compute_hash(item)

        i = 0
        while i < self.k:
            if self.bf[hash_values[i]] == 0:
                return False
            i += 1

        return True

    def get_bf(self):
        """Return the current state of the bit array"""
        return self.bf

######################## STABLE BLOOM FILTER

class StableBloomFilter:
    def __init__(self, m, d, k, p):
        """Initialize the SBF structure."""

        self.m = m  # Number of cells in the structure
        self.d = d  # Number of bits per cell
        self.k = k  # Number of hash functions
        self.p = p  # Number of cells to decrement to maintain stability
        self.max_val = (2 ** d) - 1  # Maximum value representable in d bits
        self.sbf = [0] * m  # Initialize cells to 0

    def _compute_hash(self, item):
        """Generate k independent hashes using double hashing"""
        hash_values = []

        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)

        for i in range(self.k):
            hash_values.append((h1 + i * h2) % self.m)

        return hash_values

    def check(self, item):
        """Check for potential presence of the element"""
        hash_values = self._compute_hash(item)

        i = 0
        while i < self.k:
            if self.sbf[hash_values[i]] == 0:
                return False
            i += 1

        return True

    def add(self, item):
        """Insert the element and maintain stability"""

        hash_values = self._compute_hash(item)

        # Randomly select p cells to decrement
        decrement_indices = random.sample(range(self.m), self.p)
        i = 0
        while i < self.p:
            if self.sbf[decrement_indices[i]] >= 1:
                self.sbf[decrement_indices[i]] -= 1
            i += 1

        # Update cells associated with the hashes
        i = 0
        while i < self.k:
            self.sbf[hash_values[i]] = self.max_val
            i += 1

    def get_sbf(self):
        """Return the current state of the structure"""
        return self.sbf

######################## Parameter setup and data acquisition / Experimental setup and dataset download

def create_BF_test_params(m_values, k_values):
    """Combinatorial construction of parameters for the Bloom Filter"""
    bf_params = []
    for m, k in itertools.product(m_values, k_values):
        bf_params.append({'m': m, 'k': k, 'F1': 0.0, 't': 0.0})
    return pd.DataFrame(bf_params)

def create_SBF_test_params(fps_values, m_values, k_values, max_values):
    """
    Compute parameters for the Stable Bloom Filter, including p based on the theoretical formula.
    Handles division by zero and invalid results.
    """
    sbf_params = []
    for fps, m, k, max_val in itertools.product(fps_values, m_values, k_values, max_values):
        try:
            inner_term = (1 - fps**(1/k))**(1/max_val)
            if inner_term == 0:
                p_val = None
            else:
                denp = (1 / inner_term - 1) * (1/k - 1/m)
                if denp == 0:
                    p_val = None
                else:
                    p_raw = 1 / denp
                    p_val = int(round(p_raw))
        except Exception:
            p_val = None

        sbf_params.append({
            'fps': fps,
            'm': m,
            'k': k,
            'max_val': max_val,
            'p': p_val,
            'F1': 0.0,
            't': 0.0
        })

    return pd.DataFrame(sbf_params)

######################## URL ACQUISITION FROM COMMON CRAWL

def download_and_process_urls(path, n_prova=None):
    """
    Download of the gzip file from Common Crawl, JSON record parsing,
    URL extraction and counting, decoding error handling.
    """

    # path: partial link to the .gz index file in https://data.commoncrawl.org/cc-index/collections/index.html
    #       used for accessing the URLs
    # n_prova: parameter to limit the number of processed URLs. Useful to avoid downloading and analyzing millions of URLs,
    #          e.g., during testing, debugging, or exploratory analysis on smaller subsets.

    
    start = time.perf_counter()

    BASE_URL = "https://data.commoncrawl.org/"
    url = BASE_URL + path
    print(f"Downloading {url}...")

    response = requests.get(url, stream=True)  # File download
    if response.status_code != 200:
        print('NOT DOWNLOAD!')
        return None

    urls = []
    n_errors = 0  # Errors while reading URLs
    n_processed = 0

    with gzip.GzipFile(fileobj=response.raw) as f:
        # Open compressed file: #response.raw = raw HTTP response object passed for streaming read
        for line in f:
            line = line.decode('utf-8').strip()  # Decode line from bytes to UTF-8 and strip whitespace or newline characters
            if line:
                try:
                    start_index = line.find('{')  # Find the beginning of the JSON object in the line
                    if start_index != -1:
                        json_part = line[start_index:]  # Extract the JSON substring
                        record = json.loads(json_part)  # Convert to Python dictionary
                        urls.append(record['url'])  # Save the value
                        n_processed += 1

                        # Progress status:
                        if (n_processed % 10**4) == 0:
                            print(f"Dowloaded {n_processed // 10**3} k urls")
                        
                        # To process only a limited number of URLs:
                        if n_processed == n_prova:
                            break

                except json.JSONDecodeError:
                    n_errors += 1

    stop = time.perf_counter()

    distinct_urls = set(urls)
    n_distinct = len(distinct_urls)
    n_duplicates = n_processed - n_distinct

    return {'N': n_processed,
            'Number of errors': n_errors,
            'Number of duplicates': n_duplicates,
            'Execution time': stop - start,
            'urls': urls,
            'distinct urls': distinct_urls}

######################## EMPIRICAL TESTING, RUNNING EXPERIMENTS

def BF_tests(bf_test_params, urls):
    results = []

    for index, row in bf_test_params.iterrows():
        m, k = int(row['m']), int(row['k'])
        bf = BloomFilter(m, k)
        tpos = tneg = fpos = fneg = 0
        t_start = time.perf_counter()
        ls = set()  # Set of already seen values

        for item in urls:
            if not bf.check(item):  # Item not detected by the filter
                if item in ls:
                    fneg += 1
                else:
                    tneg += 1
                bf.add(item)
            else:  # Item considered already seen by the filter
                if item in ls:
                    tpos += 1
                else:
                    fpos += 1
            ls.add(item)

        t_stop = time.perf_counter()
        t = t_stop - t_start
        F1 = (2 * tpos) / (2 * tpos + fpos + fneg) if (2 * tpos + fpos + fneg) > 0 else 0

        results.append({
            'm': m, 'k': k, 'F1': F1, 't': t,
            'tpos': tpos, 'tneg': tneg, 'fpos': fpos, 'fneg': fneg
        })

        print(f"BF - m={m}, k={k}, time={t:.4f}, F1={F1:.4f}, tpos={tpos}, tneg={tneg}, fpos={fpos}, fneg={fneg}")

    return pd.DataFrame(results)

def SBF_tests(sbf_test_params, urls):
    results = []

    for index, row in sbf_test_params.iterrows():
        fps, m, k, max_val, p = row['fps'], int(row['m']), int(row['k']), int(row['max_val']), int(row['p'])
        d = int(math.log(max_val + 1, 2))  # Bits per cell

        if p is None:
            print(f"SBF - Skipping: fps={fps}, m={m}, k={k}, max_val={max_val}, p=None")
            continue

        sbf = StableBloomFilter(m, d, k, p)
        tpos = tneg = fpos = fneg = 0
        t_start = time.perf_counter()
        ls = set()

        for item in urls:
            if not sbf.check(item):
                if item in ls:
                    fneg += 1
                else:
                    tneg += 1
            else:
                if item in ls:
                    tpos += 1
                else:
                    fpos += 1
            sbf.add(item)
            ls.add(item)

        t_stop = time.perf_counter()
        t = t_stop - t_start

        if (2 * tpos + fpos + fneg) != 0:
            F1 = (2 * tpos) / (2 * tpos + fpos + fneg) 
        else: F1 = 0

        results.append({
            'fps': fps, 'm': m, 'k': k, 'max_val': max_val, 'p': p,
            'F1': F1, 't': t, 'tpos': tpos, 'tneg': tneg, 'fpos': fpos, 'fneg': fneg
        })

        print(f"SBF - m={m}, d={d}, k={k}, p={p}, time={t:.4f}, F1={F1:.4f}, tpos={tpos}, tneg={tneg}, fpos={fpos}, fneg={fneg}")

    return pd.DataFrame(results)

def run_filter_experiments(path = "cc-index/collections/CC-MAIN-2024-42/indexes/cdx-00000.gz"):
    """ Experiment execution function
    path = compressed folder to download data from, taken from https://data.commoncrawl.org/cc-index/collections/index.html"""
    
    # Settings for generating the tables
    random.seed(42)  # set seed for reproducibility
    fps_values = [0.05, 0.01]
    m_values = [500000, 10**6, 5*10**6, 10*10**6, 40*10**6]
    k_values = [2, 4, 6, 8]
    max_values = [1, 3, 5]

    # Parameters for BF and SBF tests
    bf_test_params_df = create_BF_test_params(m_values, k_values)
    sbf_test_params_df = create_SBF_test_params(fps_values, m_values, k_values, max_values)

    # Base URL for Common Crawl
    BASE_URL = "https://data.commoncrawl.org/"
    path = path
    #n_prova = 2*10**4  # Use this number of data

    # Download and process URLs
    data_info = download_and_process_urls(path)

    if data_info:  # Check if download was successful

        # Run BF tests
        bf_results_df = BF_tests(bf_test_params_df, data_info['urls'])
        bf_results_df.to_csv('bf_results.csv', index=False)
        
        # Run SBF tests
        sbf_results_df = SBF_tests(sbf_test_params_df, data_info['urls'])
        sbf_results_df.to_csv('sbf_results.csv', index=False)

        # Select best BF parameters based on F1 and (in case of tie) complexity
        max_F1_bf = bf_results_df['F1'].max()
        bf_top = bf_results_df[bf_results_df['F1'] == max_F1_bf].copy()
        bf_top['complexity'] = bf_top['m']+bf_top['k']
        best_bf_params = bf_top.sort_values(by='complexity').head(1)
        print('Best parameters for BF:', best_bf_params)

        # Select best SBF parameters based on F1 and (in case of tie) complexity
        max_F1_sbf = sbf_results_df['F1'].max()
        sbf_top = sbf_results_df[sbf_results_df['F1'] == max_F1_sbf].copy()
        sbf_top['complexity'] = sbf_top['m']+sbf_top['k']+sbf_top['p']
        best_sbf_params = sbf_top.sort_values(by='complexity').head(1)
        
        # Best parameter configurations
        print('Best parameters for BF:', best_bf_params)
        print('Best parameters for SBF:', best_sbf_params)

        # Save basic information
        df_info = pd.DataFrame([data_info])
        df_info.drop(['urls', 'distinct_urls'], axis=1, inplace=True)  # Remove lists from the DataFrame
        df_info.to_csv('data_info.csv', index=False)

    else:
        print("Data download failed, tests not executed.")

def apply_filtering(filter_type, urls, m, k, fps=None, max_val=None, p=None):
    """
    Applies a filter (Bloom Filter - BF or Stable Bloom Filter - SBF) to the given URLs.
    Returns only the URLs considered 'new' (i.e., not duplicates) by the filter.

    Parameters:
    - filter_type: 'BF' or 'SBF'
    - urls: list of strings (URLs)
    - m: size of the filter
    - k: number of hash functions
    - fps: estimated arrival rate (only for SBF)
    - max_val: estimated maximum count (only for SBF)
    - p: number of decrements (only for SBF, optional)
    """
    filtered_urls = []

    if filter_type == "BF":
        bf = BloomFilter(m, k)
        for url in urls:
            if not bf.check(url):  # New URL
                filtered_urls.append(url)
                bf.add(url)

    elif filter_type == "SBF":
        if fps is None or max_val is None:
            raise ValueError("fps and max_val are required for SBF.")

        if p is None:
            p = max(1, int(k / 2))  # Default heuristic for p

        d = int(math.log2(max_val + 1))  # Cell depth
        sbf = StableBloomFilter(m, d, k, p)

        for url in urls:
            if not sbf.check(url):  # New URL
                filtered_urls.append(url)
            sbf.add(url)

    else:
        raise ValueError("Invalid filter type. Use 'BF' or 'SBF'.")

    return filtered_urls

# Run experiments (⚠️ this may take a long time — hours!)
run_filter_experiments()

# ↓↓↓ Applying the filter with optimal parameters ↓↓↓
# Decompression of URLs from the Common Crawl dataset
d = download_and_process_urls(path="cc-index/collections/CC-MAIN-2024-42/indexes/cdx-00000.gz")

# Applying the Stable Bloom Filter (SBF) with optimal configuration
filtered_data = apply_filtering(
    filter_type='SBF',
    urls=d['urls'],
    m=40_000_000,
    k=8,
    fps=0.05,
    max_val=1,
    p=None  # p is automatically computed if not specified
)

# Compute the exact false positive rate:
false_positive_rate = (len(d['distinct urls']) - len(filtered_data)) / len(d['distinct urls'])
print(f"False Positive Rate: {false_positive_rate:.4%}")

# Observation:
# When saving data to .txt and .duckdb files, the latter uses about half the memory
# compared to the text file, thanks to binary compression.
