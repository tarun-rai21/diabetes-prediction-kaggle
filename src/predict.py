import pandas as pd


def make_submission(model, X_test, sample_path, output_path):
    """
    Generate submission CSV file
    """

    sample = pd.read_csv(sample_path)

    # Predict probability of diabetes
    preds = model.predict_proba(X_test)[:, 1]

    sample["diagnosed_diabetes"] = preds
    sample.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
