import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing steps:
    - Handle missing values
    - Fix datatypes if needed
    - Return cleaned dataframe
    """

    # Example: Fill missing numerical values with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Example: Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df
