# Dense-3B-Arch1 Model Evaluation

> Evaluate **tiiuae/dense-3b-arch1** (3B parameters) on MMLU datasets

**Model**: [dense-3b-arch1](https://huggingface.co/tiiuae/dense-3b-arch1) | **Checkpoints**: 27 (iter_0002000 → iter_0054000)

---

## 🚀 Quick Start

### Option 1: Interactive Menu (Easiest)

```bash
cd /home2/yth/dense3b_evaluation
bash START_HERE.sh
```

### Option 2: Parallel Evaluation (Fastest - Recommended!)

```bash
# Run both evaluations simultaneously on different GPUs
bash RUN_PARALLEL.sh

# Monitor progress
bash MONITOR_PARALLEL.sh
```

### Option 3: Individual Evaluations with GPU Selection

```bash
# Fill-in-the-blank (GPU 2 recommended) - ~6-7 hours
cd "Fill-in-the-blank questions"
CUDA_VISIBLE_DEVICES=2 bash run_evaluation_fillblank_FULL.sh

# Multiple Choice (GPU 3 recommended) - ~13-27 hours  
cd "Multiple-choice questions"
CUDA_VISIBLE_DEVICES=3 bash run_evaluation_dense3b.sh
```

---

## 📂 Project Structure

```
dense3b_evaluation/
├── README.md                        # This file
├── START_HERE.sh                    # Interactive menu
├── RUN_PARALLEL.sh                  # Run both evaluations in parallel
├── MONITOR_PARALLEL.sh              # Monitor parallel progress
│
├── Multiple-choice questions/       # Standard MMLU (57 subjects, ~14K questions)
│   ├── run_evaluation_dense3b.sh       # Full evaluation
│   ├── run_evaluation_dense3b_GPU.sh   # With GPU selection
│   ├── quick_test_dense3b.sh           # Quick test
│   ├── view_results_dense3b.py         # View results
│   ├── plot_results_dense3b.py         # Generate charts
│   └── results_dense3b/                # Results (auto-created)
│
└── Fill-in-the-blank questions/     # Custom MMLU (43 subjects, 8.5K questions)
    ├── run_evaluation_fillblank_FULL.sh    # Full evaluation
    ├── run_evaluation_fillblank_GPU.sh     # With GPU selection  
    ├── run_evaluation_fillblank_BATCH.py   # Batch evaluator (20x faster!)
    ├── view_results_fillblank.py           # View results
    ├── plot_results_fillblank.py           # Generate charts
    └── results_fillblank3b_batch/          # Results (auto-created)
```

---

## 🎯 Two Evaluation Types

| Type | Dataset | Task | Speed | Time |
|------|---------|------|-------|------|
| **Multiple Choice** | cais/mmlu (57 subjects) | 4-choice questions | ~800 samples/sec | ~13-27h |
| **Fill-in-the-blank** | Custom (43 subjects) | Generative fill-in | ~10 samples/sec | ~6-7h |

---

## 💻 GPU Selection Examples

### Method 1: Command Line (Most Flexible)

```bash
# Multiple Choice on GPU 3
cd "Multiple-choice questions"
CUDA_VISIBLE_DEVICES=3 bash run_evaluation_dense3b.sh

# Fill-in-the-blank on GPU 2
cd "Fill-in-the-blank questions"
CUDA_VISIBLE_DEVICES=2 bash run_evaluation_fillblank_FULL.sh
```

### Method 2: GPU Selection Scripts

```bash
# Edit GPU_ID in scripts first
cd "Multiple-choice questions"
bash run_evaluation_dense3b_GPU.sh

cd "Fill-in-the-blank questions"
bash run_evaluation_fillblank_GPU.sh
```

---

## ⚡ Recommended GPU Assignment

**You have 8 × RTX 3090, suggested allocation**:

| GPU | Task | Model Size | Memory Needed |
|-----|------|-----------|---------------|
| GPU 0 | 500M Multiple Choice | 500M | ~2-4 GB |
| GPU 1 | 500M Fill-in-the-blank | 500M | ~2-4 GB |
| GPU 2 | **3B Fill-in-the-blank** | **3B** | **~8-12 GB** |
| GPU 3 | **3B Multiple Choice** | **3B** | **~8-12 GB** |
| GPU 4 | 1B Multiple Choice | 1B | ~4-6 GB |
| GPU 5 | 1B Fill-in-the-blank | 1B | ~4-6 GB |
| GPU 6-7 | Reserved | - | - |

---

## 📊 View Results

```bash
# Multiple Choice
cd "Multiple-choice questions"
python3 view_results_dense3b.py
python3 plot_results_dense3b.py

# Fill-in-the-blank
cd "Fill-in-the-blank questions"
python3 view_results_fillblank.py
python3 plot_results_fillblank.py
```

---

## ⏱️ Time Estimates

| Task | Samples | Speed | Time/checkpoint | Total (27 checkpoints) |
|------|---------|-------|----------------|----------------------|
| Fill-in-the-blank | 8,500 | ~10/sec | ~14 min | **~6-7 hours** |
| Multiple Choice | ~14,079 | ~800/sec | ~30 min | ~13-27 hours |

**Note**: 3B model is **6x larger** than 500M, may be slightly slower per sample

---

## 🆚 Comparison: 500M vs 1B vs 3B

| Metric | Dense-500M | Dense-3B | Dense-3B (Expected) |
|--------|------------|----------|---------------------|
| Parameters | 500M | 1B | **3B (6x of 500M)** |
| MC MMLU | ~35-45% | ~40-55% | **~45-60%** |
| Fill-in-blank | ~2-5% | ~5-10% | **~8-15%** |
| GPU Memory | ~2-4 GB | ~4-6 GB | **~8-12 GB** |

---

## 🔧 Common Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPUs in real-time
watch -n 1 nvidia-smi

# Background run
cd "Fill-in-the-blank questions"
CUDA_VISIBLE_DEVICES=2 nohup bash run_evaluation_fillblank_FULL.sh > eval.log 2>&1 &
tail -f eval.log
```

---

## 📞 Quick Reference

| What | How |
|------|-----|
| **Start everything** | `bash START_HERE.sh` |
| **Parallel run** | `bash RUN_PARALLEL.sh` |
| **MC on GPU 3** | `CUDA_VISIBLE_DEVICES=3 bash run_evaluation_dense3b.sh` |
| **FB on GPU 2** | `CUDA_VISIBLE_DEVICES=2 bash run_evaluation_fillblank_FULL.sh` |
| **View results** | `python3 view_results_*.py` |
| **Generate plots** | `python3 plot_results_*.py` |

---

**Ready to go!** 🚀

Model: **tiiuae/dense-3b-arch1** (3B parameters - largest model!)

