# Guide: Transferring MENTOR to a High-Performance System

This guide outlines how to migrate the project to your powerful workstation/server and train on the real **Epinions** (664,824 ratings) and **Ciao** (284,086 ratings) datasets.

---

## 1. Quick Transfer Steps

### Step 1.1: Copy the Project
Copy the `mentor/` directory to your target machine:
```bash
# On your target machine, enter the directory:
cd path/to/mentor
```

### Step 1.2: Set Up Python Environment (Python 3.10+)
```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify installation with the unit tests
python run_tests.py
```
*(All 7 tests should print `OK`).*

---

## 2. Downloading the Real Datasets

We have created an automated downloader that pulls the verified academic mirrors of both datasets:

```bash
python data/download_real_data.py
```

This will automatically create:
- `data/epinions_clean.csv` (664,824 ratings, 49,289 users, 139,738 items)
- `data/ciao_clean.csv` (284,086 ratings, 7,375 users, 106,797 items)

---

## 3. Running Benchmarks on Real Data

### Option A: Quick Sanity Test (20% of Real Epinions)
Before launching the full multi-hour benchmark, test with a 20% sample:
```bash
python experiments/run_real_benchmarks.py --dataset epinions --sample_fraction 0.2
```

### Option B: Full Real Epinions Benchmark (100% Data)
Runs the complete paper benchmark on all 664,824 ratings:
```bash
python experiments/run_real_benchmarks.py --dataset epinions --sample_fraction 1.0
```

### Option C: Full Real Ciao Benchmark (100% Data)
Runs on all 284,086 Ciao ratings:
```bash
python experiments/run_real_benchmarks.py --dataset ciao --sample_fraction 1.0
```

### Option D: Custom Privacy Budget Sweep
```bash
python experiments/run_real_benchmarks.py --dataset epinions --epsilons 0.1 0.5 1.0
```

---

## 4. Output Artifacts

After execution, all metrics and plots are automatically generated in the `results/` folder:
- `results/real_epinions_results.csv`
- `results/real_epinions_plot.png`
- `results/real_ciao_results.csv`
- `results/real_ciao_plot.png`

---

## 5. Performance Tips for Powerful Machines

1. **Multi-threading with OpenBLAS / MKL**:
   NumPy matrix dot products will automatically utilize all CPU cores. Ensure `OMP_NUM_THREADS` is set if using Linux/server environments:
   ```bash
   export OMP_NUM_THREADS=8  # or your CPU core count
   ```
2. **RAM Requirements**:
   - Epinions (full): ~4 GB to 8 GB RAM during sparse dictionary mapping.
   - Ciao (full): ~2 GB to 4 GB RAM.
