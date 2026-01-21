# pruning-algorithms-research-project

This project investigates the efficacy of pruning algorithms such as Iterative Magnitude Pruning (IMP) and Signle-Shot Network Pruning (SNIP) in reducing the size of a ResNet-18 model while trying to maintain classification accuracy on the Oxford Flowers-102 fine-grained classification dataset.

The framework is designed to be extensible for other pruning methods and experiments.

## Main Results

*(Figure 1: Test Accuracy vs. Parameter Count. The dot represents the unpruned ResNet18 model (~11.2M params). The lines shows the performance retention of IMP/SNIP across sparsity levels)*

## Setup

The project uses Conda to manage its environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/frustal/pruning-algorithms-research-project.git
    cd pruning-algorithms-research-project
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate pruning_env
    ```

3.  **Dataset:** The Flowers-102 dataset will be automatically downloaded to a `data/` directory the first time you run an experiment.

## Usage

Experiments are managed through YAML configuration files located in the `configs/` directory.

### Running Experiments

To run an experiment, use the `train.py` script and specify a configuration file.

**1. Default Training (Dense Model)**

This runs a standard training loop without any pruning.
```bash
python train.py --config configs/default.yaml
```

**2. Iterative Magnitude Pruning (IMP)**

This runs the full IMP pipeline:
- **Phase 1:** Train a dense model to convergence.
- **Phase 2:** Prune the model globally based on weight magnitude to various sparsity targets.
- **Phase 3:** Retrain the pruned model for a few epochs.

```bash
python train.py --config configs/imp.yaml
```

**Debug Mode**

For quick tests, you can use the `--debug` flag. This will run the experiment for only one epoch on a small subset of the data.
```bash
python train.py --config configs/imp.yaml --debug
```

### Experiment Outputs

-   **Trained Models:** Model weights (`.pth` files) are saved in `output/models/`.
-   **Logs & Metrics:** Results such as accuracy, sparsity, and parameter count are logged to `output/logs/<experiment_name>/metrics.csv`.

### Plotting Results

After running your experiments, you can generate a comparative plot of test accuracy versus model size (in millions of parameters).

The `plot_results.py` script reads the `metrics.csv` files from the specified experiments.

```bash
# Example: Compare the default run with the IMP run
python plot_results.py --experiments default imp_baseline
```

This will save a `final_results.png` image in the root directory.

### Cleaning Outputs

A utility script is provided to clear all generated logs and models, allowing for a fresh run.

```bash
python clear_outputs.py
```

## Project Structure

```
.
├── configs/                # YAML configuration files for experiments
│   ├── default.yaml        # Config for standard dense training
│   └── imp.yaml            # Config for Iterative Magnitude Pruning
├── src/                    # Source code
│   ├── data.py             # Dataloaders for the Flowers-102 dataset
│   ├── model.py            # ResNet-18 model definition
│   ├── utils.py            # Helper functions (seeding, logging)
│   └── methods/            # Implementations of training/pruning methods
│       ├── default.py      # Standard training loop
│       └── imp.py          # IMP training, pruning, and retraining loop
├── train.py                # Main script to launch experiments
├── plot_results.py         # Script to generate comparison plots
├── environment.yaml        # Conda environment definition
└── clear_outputs.py        # Utility to clear output directories
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
