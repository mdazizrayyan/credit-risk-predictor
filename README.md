 Credit risk prediction using Logistic Regression with SHAP explainability, deployed on Hugging Face Spaces
# Credit Risk Predictor

A machine learning system that predicts whether a loan applicant is a credit risk (likely to default), built on the UCI German Credit dataset.

🟢 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/azizrayyan/credit-risk-predictor)**

## Results
| Metric | Value |
|--------|-------|
| AUC | 0.80 |
| Recall (default class) | 0.80 |
| Accuracy | 0.75 |

## Problem & Approach
Credit risk datasets are imbalanced — 70% good loans, 30% defaults. A naive model predicting "good" for everyone achieves 70% accuracy while being completely useless. This project optimizes for **recall on the default class** — catching bad loans matters more than overall accuracy in lending.

## Key Decisions
**Logistic Regression over Random Forest** — after comparing both models, logistic regression with `class_weight='balanced'` achieved 0.80 recall on the default class vs 0.37 for random forest. The simpler model won on the metric that matters.

**SHAP Explainability** — financial models require interpretability. SHAP values show which features drive each prediction. Top findings:
- High `credit_amount` and long `duration` are the strongest predictors of default
- High `installment_rate` (large % of income going to repayments) increases risk
- These align with domain intuition — larger, longer loans on tight budgets default more

## What I Would Improve in Production
- Retrain on more recent, larger dataset (German Credit is from 1994)
- Add model monitoring for data drift
- Use SHAP values per-prediction in the UI (not just global feature importance)
- Replace one-hot encoding with target encoding for high-cardinality categoricals

## Tech Stack
- Python, scikit-learn, pandas, SHAP
- Gradio for the demo interface
- Hugging Face Spaces for deployment

## Dataset
[UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — 1000 loan applicants, 20 features, binary classification (good/bad credit risk)
