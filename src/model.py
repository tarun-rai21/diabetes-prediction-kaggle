from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train XGBoost model and return trained model + validation score
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)

    print(f"Validation AUC: {auc:.4f}")

    return model, auc
