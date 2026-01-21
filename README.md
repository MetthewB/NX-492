## Semester Project at the Mathis Group for Computational Neuroscience and AI - Parallel Feedback & Neural Limb Control

This repository implements a biologically plausible neural limb controller to investigate hierarchical motor control. It reproduces and extends the findings of the 2024 study **"Parallel feedback processing for voluntary control: knowledge of spinal feedback in motor cortex"** by Guang et al..

### 🎯 Core Objective

How does the nervous system coordinate fast, low-level spinal reflexes with slower, high-level cortical responses?

This project simulates a musculoskeletal arm model driven by a Recurrent Neural Network (RNN) cortex and a spinal reflex loop. The goal is to demonstrate "reciprocal reduction": the hypothesis that the motor cortex learns to "offload" control to spinal reflexes when those reflexes are strengthened by high background loads.

---

## 📊 Key Findings

By training the model for 3,000 epochs using differentiable physics, we successfully replicated the core biological predictions:

* **Robust Stabilization:** The controller converges to a robust policy that acts as a steep energy well, stabilizing the 2-DOF arm against random perturbations.
* **Impedance Control:** The model learns to completely compensate for high background loads, likely through co-contraction strategies that increase limb stiffness.
* **Reciprocal Reduction:** Crucially, we observe that the motor cortex reduces its activity during high-load conditions. It effectively "trusts" the strengthened spinal reflexes to handle stability, validating the efficiency hypothesis of hierarchical motor control.

> **Explore the Data:** For a detailed analysis of these results, see the **Summary** section at the end of `script/main.ipynb` or view the generated plots in `results/100_epochs/` and `results/3000_epochs/`.

---

## 🚀 Getting Started

### Prerequisites

The code is tested on **Python 3.11.11**. You will need the following libraries:

* PyTorch
* Gymnasium
* NumPy
* Matplotlib

### Installation

You can install the necessary dependencies via pip:

```bash
pip install torch gymnasium numpy matplotlib
```

### Recommended Workflow

1. **Main Simulation:** Open `script/main.ipynb`. This is the primary entry point.
* **Context:** Explains the scientific background and model architecture.
* **Code:** Runs the full differentiable physics pipeline and training loop.
* **Analysis:** Generates the cost evolution and trajectory comparison plots.

2. **Environment Mechanics:** Check `script/MyoElbowPose2D6M.py` to see the custom Gym environment, which implements the 2-DOF arm physics, Hill-type muscle dynamics, and moment arm geometry.

3. **Foundational Concepts:** To understand the specific reflex logic (Ia/Ib afferents, reciprocal inhibition), explore `script/stretch_reflex_toy.ipynb`. This notebook builds the reflex model complexity step-by-step.
4. **Alternative Approaches:** See `script/flexible_walker.ipynb` for an implementation based on the secondary reference (Ramadan et al.). Note: This notebook is experimental and not fully functional, but serves to illustrate alternative hierarchical control strategies.



---

## 📂 Repository Structure

```text
.
├── README.md                      # Project overview and documentation
├── architecture.md                # Detailed technical specification of the model architecture
├── Guang_ParallelFeedback.pdf     # Primary reference article (Guang et al., 2024)
├── Ramadan_FlexibleWalker.pdf     # Secondary reference article (Ramadan et al.)
│
├── script/                        # Codebase and Jupyter Notebooks
│   ├── main.ipynb                 # MAIN FILE: Context, Code, and Analysis
│   ├── MyoElbowPose2D6M.py        # Custom Gymnasium environment (Physics & Biomechanics)
│   ├── stretch_reflex_toy.ipynb   # Progressive reflex logic experiments
│   ├── flexible_walker.ipynb      # Alternative hierarchical strategy (Ramadan et al.)
│   └── intermediate_version.ipynb # Archived development steps
│
├── results/                       # Generated plots and metrics
│   ├── 100_epochs/                # Early-stage training plots
│   ├── 3000_epochs/               # Fully converged training plots
│   └── toy_model/                 # Outputs from reflex toy experiments
│
└── images/                        # Static assets
    └── flexible_walker.png        # Diagram of the flexible walker model
```

---

## 📚 References

```bibtex
@article{guang2024parallel,
      title={Parallel feedback processing for voluntary control: knowledge of spinal feedback in motor cortex}, 
      author={Hui Guang and Joseph Y. Nashed and J. Andrew Pruszynski and Hari Teja Kalidindi and Frederic Crevecoeur and Kevin P. Cross and Gunnar Blohm and Stephen H. Scott},
      journal={bioRxiv},
      year={2024},
      doi={10.1101/2024.05.12.593756},
      url={https://www.biorxiv.org/content/10.1101/2024.05.12.593756v1.full},
      publisher={Cold Spring Harbor Laboratory}
}

@article{ramadan2022neuromuscular,
      title={A neuromuscular model of human locomotion combines spinal reflex circuits with voluntary movements},
      author={Rachid Ramadan and Hartmut Geyer and John Jeka and Gregor Schöner and Hendrik Reimann},
      journal={Scientific Reports},
      year={2022},
      publisher={Nature Publishing Group},
      url={https://www.nature.com/articles/s41598-022-11102-1},
      doi={10.1038/s41598-022-11102-1}
}
```