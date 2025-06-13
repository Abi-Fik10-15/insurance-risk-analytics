# End-to-End Insurance Risk Analytics & Predictive Modeling

## Project Overview
This project performs insurance risk analytics by building predictive models from claims data. It includes data cleaning, training an XGBoost classifier, and evaluating model performance.

## Quick Start

```bash
# Clone repo
git clone <repo-url>
cd insurance-risk-analytics

# Setup environment
conda env create -f environment.yml
conda activate insurance-risk

# Initialize DVC and pull data
dvc pull
dvc repro
