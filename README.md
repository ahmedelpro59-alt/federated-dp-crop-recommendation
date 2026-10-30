# Federated learning with differential-privacy noise for crop recommendation

Code for the manuscript **"Quantifying the Privacy–Accuracy Trade-off of Differentially Private Federated Learning for Crop Recommendation under IID and Non-IID Data Distributions"**.

The experiment compares, on a soil-nutrient crop dataset:

- centralized training,
- federated averaging (FedAvg) with five simulated clients,
- FedAvg with client-update clipping and Gaussian noise (DP-FedAvg style) at noise multipliers z = 0.1, 0.5, 1.0 and 2.0,

under IID and Dirichlet label-skewed (non-IID, α = 0.5 and α = 0.1) client partitions, with five random seeds.
The model is a linear softmax classifier implemented from scratch in NumPy.

## Data

The dataset is **not included** in this repository. Download *Crop Recommender Dataset with Soil Nutrients* from Kaggle:
<https://www.kaggle.com/datasets/manikantasanjayv/crop-recommender-dataset-with-soil-nutrients>
and save it as `dataset.csv`. Check the dataset page for its terms of use.

Expected format: 11 numeric soil attributes (N, P, K, pH, EC, S, Cu, Fe, Mn, Zn, B) followed by the crop label as the **last column**
(620 samples, six crops in the file used for the paper).

## How to run

```bash
pip install -r requirements.txt
python fl_dp_experiment.py dataset.csv        # writes results/results_table.csv, results_raw.csv and two figures
python analysis_paired_tests.py results/results_raw.csv   # paired differences and IID vs non-IID t-tests
```

It also runs in Google Colab: upload `fl_dp_experiment.py` and `dataset.csv`, then run `!python fl_dp_experiment.py dataset.csv` in a code cell.

## Main settings (see the top of `fl_dp_experiment.py`)

| Setting | Value |
|---|---|
| Clients / rounds / local epochs | 5 / 50 / 2 |
| Learning rate / batch size | 0.1 / 32 |
| Clipping norm C | 1.0 |
| Noise multiplier z | 0 (clipping only), 0.1, 0.5, 1.0, 2.0 |
| Seeds | 0–4 (each seed: new 80/20 stratified split and new client partition) |

## Results included

The `results/` folder contains the outputs used in the paper: `results_table.csv` (mean and SD over seeds), `results_raw.csv` (every run) and the two figures.

## Important notes

- Privacy is reported through the **noise multiplier z**. Noise is added by the server to the average of clipped client updates (client-level, central DP). **No formal (ε, δ) guarantee is computed.**
- Plain FedAvg uses size-weighted averaging, whereas the clipping and DP variants use an unweighted mean of clipped updates.
- The experiment uses a small dataset, a linear model and simulated clients, so the results should not be generalized without further testing.

## Citation

If you use this code, please cite the manuscript (add the full reference here after publication).
