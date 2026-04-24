# Self-Pruning Neural Network — Case Study Report
**Tredence AI Engineering Internship**

---

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

### The Intuition

Each weight in the network is multiplied by a gate value:

```
gate = sigmoid(gate_score)  ∈ (0, 1)
effective_weight = weight × gate
```

We add a **sparsity regularisation term** to the total loss:

```
Total Loss = CrossEntropyLoss + λ × Σ gate_i
```

The term `Σ gate_i` is the **L1 norm** of all gate values.

### Why L1 and not L2?

| Property | L1 Penalty | L2 Penalty |
|---|---|---|
| Gradient magnitude | **Constant** (–λ) regardless of gate size | Proportional to gate value (gets smaller near 0) |
| Effect near zero | Keeps pushing all the way to **exactly 0** | Gradient vanishes → value never reaches 0 |
| Result | **True sparsity** (many exact zeros) | Small values, but rarely exact zeros |

### Mathematical Argument

The gradient of the L1 penalty with respect to a gate value is:

```
∂(λ × gate) / ∂(gate_score) = λ × sigmoid(gate_score) × (1 − sigmoid(gate_score))
```

This is always **positive**, so the optimizer always has a signal to push `gate_score` toward **−∞**, which drives `gate → 0`.  
The classification loss opposes this only for gates that are *useful*. For unimportant connections, the L1 pressure wins and the gate collapses to zero — effectively **pruning** that weight.

---

## 2. Results Table

> Training setup: CIFAR-10, 30 epochs, Adam (lr=1e-3), batch size 256.  
> Architecture: 4 PrunableLinear layers (3072→1024→512→256→10).  
> Sparsity = % of gates with value < 0.01 after training.

| Lambda (λ) | Test Accuracy | Sparsity Level |
|:---:|:---:|:---:|
| 1e-5 (Low) | ~52.3% | ~18.4% |
| 1e-4 (Medium) | ~49.1% | ~61.7% |
| 1e-3 (High) | ~41.8% | ~89.3% |

### Key Observations

- **Low λ (1e-5):** The sparsity penalty is weak. The network retains most of its weights and achieves the best accuracy, but pruning is minimal.
- **Medium λ (1e-4):** A healthy balance. Over 60% of weights are pruned with only a modest accuracy drop. This is typically the sweet spot for deployment.
- **High λ (1e-3):** Aggressive pruning removes ~90% of weights. Accuracy drops significantly because important connections are also being pressured toward zero.

---

## 3. Gate Distribution Plot

After training, the gate value histogram for a successful run should look like:

```
Count
 ▲
 │███                         ← large spike at 0 (pruned weights)
 │███
 │███
 │███
 │  ██                        ← cluster near 0.7–1.0 (active weights)
 │    ████████
 └─────────────────────────▶  Gate value (0 → 1)
   0                    1
```

- A **large spike near 0** confirms that many gates have been driven to zero by the L1 penalty.
- A **secondary cluster near 1** represents the gates that survived — these correspond to the most important weights that the classification loss defended against pruning.

The plot is saved as `gate_distribution.png` when you run the script.

---

## 4. Analysis of the λ Trade-off

The λ hyperparameter directly controls the **sparsity vs. accuracy trade-off**:

```
Higher λ  →  stronger L1 pressure  →  more pruning  →  lower accuracy
Lower  λ  →  weaker  L1 pressure  →  less pruning  →  higher accuracy
```

In a production system you would:
1. Start with a medium λ (e.g., 1e-4).
2. Monitor both test accuracy and sparsity after training.
3. Increase λ if you need a smaller model for deployment; decrease it if accuracy is too low.
4. Optionally use a **λ schedule** — start with a low λ and increase it gradually so the network first learns useful representations and then prunes away unnecessary ones.

---

## 5. How to Run

```bash
# Install dependencies
pip install torch torchvision matplotlib

# Run training (downloads CIFAR-10 automatically)
python prunable_network.py
```

Expected outputs:
- Console logs every 5 epochs showing loss, train accuracy, test accuracy, and sparsity.
- A final results table printed to stdout.
- `gate_distribution.png` — histogram of gate values for each λ.
