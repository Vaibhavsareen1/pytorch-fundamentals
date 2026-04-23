# pytorch-fundamentals

PyTorch-focused repository of from-scratch implementations of fundamental deep learning building blocks. Emphasis on understanding through experiments, analysis, and validation.

---

## 🐍 Python Version

Python 3.12.3

---

## 📁 Project Structure (Planned)

```bash
pytorch-fundamentals/
├── src/            # core implementations
├── notebooks/      # explanations & intuition
├── experiments/    # controlled experiments
├── tests/          # validation
├── benchmarks/     # comparison with PyTorch
└── requirements.txt
```

---

## 📌 Repository Structure Explained

### `src/`
Contains from-scratch implementations of fundamental deep learning components. This is the core of the repository, focused on clean, minimal, and modular implementations without experimental or exploratory code.

### `notebooks/`
Used for exploration, intuition building, and analysis of concepts implemented in `src/`. Includes derivations, comparisons, and experiments to better understand behavior and design choices.

### `experiments/`
Houses structured experiments designed to evaluate the impact of different design choices, such as optimizer behavior, initialization strategies, and activation functions.

### `tests/`
Contains validation logic to ensure correctness of implementations, often by comparing behavior against PyTorch equivalents.

### `benchmarks/`
Focused on performance and behavioral comparisons between custom implementations and PyTorch’s native implementations.

### `requirements.txt`
Lists dependencies required to run the implementations, experiments, and notebooks.

---

## 📌 Philosophy

- From-scratch implementations using PyTorch primitives  
- Clear separation: code vs experiments vs validation  
- Minimal, explicit, and readable implementations  
- Validate against PyTorch where possible  
- Focus on understanding over optimization  