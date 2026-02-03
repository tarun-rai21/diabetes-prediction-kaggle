import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature preparation for modeling
    """

    # Drop ID column if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Convert boolean to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Encode categorical columns using one-hot encoding
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
