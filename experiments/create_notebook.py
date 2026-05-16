import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Latent Potential Demand Estimation\n",
                "\n",
                "In this notebook, we walk through the methodology of estimating the latent maximum monthly volume potential for the month of January 2026. Since we do not have a true 'target variable' (ground truth), we approach this as a **left-censored demand problem**.\n",
                "\n",
                "## Methodology\n",
                "1. **Load Gold Features**: We use the pre-aggregated `model_input.parquet` which contains our POI data, historical transactions, and censoring flags.\n",
                "2. **Censoring Detection**: We identify outlets that hit a 'plateau' (very low variance, capped deliveries) or had high stockout/return ratios.\n",
                "3. **Peer Group Quantile Uplift**: For constrained outlets, we override their historical volume with the **90th percentile** of their peer group (same Size, Type, and Distributor)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "\n",
                "# Load the Gold data\n",
                "df = pd.read_parquet('../data/gold/model_input.parquet')\n",
                "print(f'Loaded {len(df)} outlets for modeling.')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Visualizing the Censoring Problem\n",
                "Outlets with extremely low Coefficient of Variation (CV) are suspicious. In the FMCG sector, perfectly flat sales indicate an artificial cap (like a rigid credit limit or strict delivery allocation), not true demand."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 5))\n",
                "sns.histplot(df['cv_volume'], bins=50, kde=True, color='purple')\n",
                "plt.axvline(0.15, color='red', linestyle='--', label='Censoring Threshold (CV < 0.15)')\n",
                "plt.title('Distribution of Volume Variance (CV)')\n",
                "plt.xlabel('Coefficient of Variation')\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Peer Group Analysis\n",
                "To estimate what a constrained outlet *could* have sold, we group them by similar characteristics: `Type`, `Size`, and `Distributor`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "peer_groups = df.groupby(['Outlet_Size_Score', 'Distributor_ID'])['peak_volume'].agg(['mean', 'median', lambda x: np.percentile(x, 90)])\n",
                "peer_groups.rename(columns={'<lambda_0>': '90th_Percentile'}, inplace=True)\n",
                "peer_groups.head(10)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Estimating Latent Potential\n",
                "We calculate the estimated potential as follows:\n",
                "- **Unconstrained (Normal):** Base estimate = Recent 3-month average + Seasonality impact.\n",
                "- **Constrained (Censored):** Potential = Max(Base, Peak Volume, Peer Group 90th Percentile)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Base Estimate (Recent 3m avg adjusted for January seasonality)\n",
                "seas_adj = 1.0 + (df['Jan_Seasonality_Score'].fillna(0) * 0.1)\n",
                "base_est = df['recent_3m_avg'] * seas_adj\n",
                "\n",
                "# Peer Group 90th Percentile Ceiling\n",
                "peer_est = df['Peer_Group_90th_Vol'].fillna(base_est)\n",
                "peak_est = df['peak_volume'].fillna(base_est)\n",
                "\n",
                "# Censoring Condition\n",
                "is_censored = (df['Is_Plateaued'] == 1) | (df['is_stockout_censored'] == 1) | (df['is_high_return_censored'] == 1)\n",
                "\n",
                "df['Latent_Potential'] = np.where(\n",
                "    is_censored,\n",
                "    np.maximum.reduce([base_est * 1.1, peer_est, peak_est * 1.05]),\n",
                "    np.maximum.reduce([base_est, peak_est * 0.9])\n",
                ")\n",
                "\n",
                "# Global Market Growth Factor for 2026\n",
                "df['Latent_Potential'] = df['Latent_Potential'] * 1.05\n",
                "\n",
                "print(f\"Average Latent Potential: {df['Latent_Potential'].mean():.2f} L\")\n",
                "print(f\"Average Historical Peak: {df['peak_volume'].mean():.2f} L\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Distribution Shift (Observed vs Potential)\n",
                "Notice how the Latent Potential distribution shifts slightly to the right, reflecting the uncapping of constrained demand."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 5))\n",
                "sns.kdeplot(df['peak_volume'].clip(upper=300), label='Historical Peak Volume', fill=True)\n",
                "sns.kdeplot(df['Latent_Potential'].clip(upper=300), label='Latent Potential (Jan 2026)', fill=True)\n",
                "plt.title('Demand Distribution: Observed Peak vs. Latent Potential')\n",
                "plt.xlabel('Volume (Liters) - Clipped at 300L for visualization')\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare final output\n",
                "output = df[['Outlet_ID', 'Latent_Potential']].copy()\n",
                "output['Latent_Potential'] = output['Latent_Potential'].round(2)\n",
                "output.to_csv('../output/insightai_predictions.csv', index=False)\n",
                "print(\"Saved final predictions to output/insightai_predictions.csv\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks/Model_Training_Walkthrough.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated.")
