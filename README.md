# Student Degree Outcome Prediction Dashboard

This is a clean Streamlit deployment package for student completion, stopping, and shifting analysis.

## Files

- `app.py` - Streamlit dashboard
- `train_models.py` - offline/online model training script
- `preprocessing.py` - shared cleaning and feature engineering functions
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version for Streamlit Cloud
- `models/` - generated model files after training
- `data/` - optional default dataset location

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional Offline Training

Place your Excel file here:

```text
data/StudProfile.xlsx
```

Then run:

```bash
python train_models.py
```

This creates:

```text
models/outcome_model.pkl
models/feature_list.pkl
models/model_metadata.json
```

## Streamlit Cloud / AuthKit Deployment

Upload the repository to GitHub and set the main file path to:

```text
app.py
```

You may upload the dataset inside the app sidebar instead of placing it in GitHub.
