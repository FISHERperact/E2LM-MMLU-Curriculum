# Dense-1B-Arch1 Model Evaluation

> Evaluate **tiiuae/dense-1b-arch1** (1B parameters) on MMLU datasets

**Model**: [dense-1b-arch1](https://huggingface.co/tiiuae/dense-1b-arch1) | **Checkpoints**: 27 (iter_0002000 → iter_0054000)

---

## 🚀 Quick Start

### Option 1: Interactive Menu (Easiest)

```bash
cd /home2/yth/dense1b_evaluation
bash START_HERE.sh
```

### Option 2: Parallel Evaluation (Fastest - Recommended!)

```bash
# Run both evaluations simultaneously on different GPUs
bash RUN_PARALLEL.sh

# Monitor progress
bash MONITOR_PARALLEL.sh
```

### Option 3: Individual Evaluations

```bash
# Fill-in-the-blank (GPU 6) - ~6-7 hours
cd "Fill-in-the-blank questions"
bash run_evaluation_fillblank_GPU.sh

# Multiple Choice (GPU 7) - ~13-27 hours  
cd "Multiple-choice questions"
bash run_evaluation_dense1b_GPU.sh
```

---

## 📂 Project Structure

```
dense1b_evaluation/
├── README.md                        # This file
├── START_HERE.sh                    # Interactive menu
├── RUN_PARALLEL.sh                  # Run both evaluations in parallel
├── MONITOR_PARALLEL.sh              # Monitor parallel progress
│
├── Multiple-choice questions/       # Standard MMLU (57 subjects, ~14K questions)
│   ├── run_evaluation_dense1b.sh       # Full evaluation
│   ├── run_evaluation_dense1b_GPU.sh   # With GPU selection
│   ├── quick_test_dense1b.sh           # Quick test (3 checkpoints)
│   ├── view_results_dense1b.py         # View results
│   ├── plot_results_dense1b.py         # Generate charts
│   └── results_dense1b/                # Results (auto-created)
│
└── Fill-in-the-blank questions/     # Custom MMLU (43 subjects, 8.5K questions)
    ├── run_evaluation_fillblank_FULL.sh    # Full evaluation
    ├── run_evaluation_fillblank_GPU.sh     # With GPU selection
    ├── run_evaluation_fillblank_BATCH.py   # Batch evaluator (20x faster!)
    ├── view_results_fillblank.py           # View results
    ├── plot_results_fillblank.py           # Generate charts
    └── results_fillblank1b_batch/          # Results (auto-created)
```

---

## 🎯 Two Evaluation Types

| Type | Dataset | Task | Speed | Time |
|------|---------|------|-------|------|
| **Multiple Choice** | cais/mmlu (57 subjects) | 4-choice questions | ~800 samples/sec | ~13-27h |
| **Fill-in-the-blank** | Custom (43 subjects) | Generative fill-in | ~10 samples/sec | ~6-7h |

---

## 💻 GPU Selection

### Method 1: Command Line (Most Flexible)

```bash
# Multiple Choice on GPU 7
cd "Multiple-choice questions"
CUDA_VISIBLE_DEVICES=7 bash run_evaluation_dense1b.sh

# Fill-in-the-blank on GPU 6
cd "Fill-in-the-blank questions"
CUDA_VISIBLE_DEVICES=6 bash run_evaluation_fillblank_FULL.sh
```

### Method 2: GPU Selection Scripts (Easier)

```bash
# Edit GPU_ID in the script (currently GPU 7)
cd "Multiple-choice questions"
bash run_evaluation_dense1b_GPU.sh

# Edit GPU_ID in the script (currently GPU 6)
cd "Fill-in-the-blank questions"
bash run_evaluation_fillblank_GPU.sh
```

### Method 3: Parallel Script (Automatic)

```bash
# Automatically assigns: MC→GPU 0, FB→GPU 1
bash RUN_PARALLEL.sh
```

---

## 📊 View Results

### Multiple Choice

```bash
cd "Multiple-choice questions"
python3 view_results_dense1b.py
python3 plot_results_dense1b.py
```

### Fill-in-the-blank

```bash
cd "Fill-in-the-blank questions"
python3 view_results_fillblank.py
python3 plot_results_fillblank.py
```

---

## ⚡ Background Execution

### Recommended: Run in background and close terminal

```bash
# Fill-in-the-blank (GPU 6)
cd "Fill-in-the-blank questions"
nohup bash run_evaluation_fillblank_GPU.sh > eval_fb.log 2>&1 &

# Multiple Choice (GPU 7)
cd "Multiple-choice questions"
nohup bash run_evaluation_dense1b_GPU.sh > eval_mc.log 2>&1 &

# Monitor
tail -f eval_fb.log  # or eval_mc.log
watch -n 1 nvidia-smi
```

---

## 🔧 Common Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPUs in real-time
watch -n 1 nvidia-smi

# Check running processes
ps aux | grep run_evaluation

# Stop evaluation
kill <PID>

# View progress
ls -d Fill-in-the-blank\ questions/results_fillblank1b_batch/iter_* | wc -l
ls -d Multiple-choice\ questions/results_dense1b/iter_* | wc -l
```

---

## ⏱️ Time Estimates

| Task | Samples | Speed | Time/checkpoint | Total (27 checkpoints) |
|------|---------|-------|----------------|----------------------|
| Fill-in-the-blank | 8,500 | ~10/sec | ~14 min | **~6-7 hours** |
| Multiple Choice | ~14,079 | ~800/sec | ~30 min | ~13-27 hours |

**Both in parallel**: ~13-27 hours total (fill-in-the-blank finishes first)

---

## 📋 Requirements

- **lm-evaluation-harness**: Installed at `/home2/yth/lm-evaluation-harness-competition`
- **Python 3.8+** with PyTorch, transformers, datasets, tqdm
- **GPU**: 8GB+ VRAM recommended (you have 8 × RTX 3090 ✅)
- **Datasets**: Will auto-download on first run

---

## 🎓 Tips

1. **Start with parallel evaluation**: `bash RUN_PARALLEL.sh`
2. **Run in background**: Use `nohup` and `&` to run overnight
3. **Monitor GPU**: Keep `watch -n 1 nvidia-smi` running
4. **Check logs regularly**: `tail -f *.log`
5. **Fill-in-the-blank is faster**: Runs ~6-7 hours vs 13-27 hours

---

## 🆚 Comparison with 500M Model

| Metric | Dense-500M | Dense-1B (Expected) |
|--------|------------|---------------------|
| Parameters | 500M | 1B (2x) |
| Multiple Choice MMLU | ~35-45% | ~40-55% |
| Fill-in-the-blank | ~2-5% | ~5-10% |

---

## 📞 Quick Reference

| What | How |
|------|-----|
| **Start everything** | `bash START_HERE.sh` |
| **Parallel run** | `bash RUN_PARALLEL.sh` |
| **Monitor parallel** | `bash MONITOR_PARALLEL.sh` |
| **MC on GPU 7** | `cd "Multiple-choice questions" && bash run_evaluation_dense1b_GPU.sh` |
| **FB on GPU 6** | `cd "Fill-in-the-blank questions" && bash run_evaluation_fillblank_GPU.sh` |
| **View results** | `python3 view_results_*.py` |
| **Generate plots** | `python3 plot_results_*.py` |
| **Check GPU** | `nvidia-smi` |

---

**Ready to go!** 🚀

Choose your method and start evaluation!
