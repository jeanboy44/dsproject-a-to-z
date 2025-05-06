import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # 데이터 불러오기
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # LightGBM 데이터셋 생성
    print("LightGBM 데이터셋 생성...")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 파라미터 지정
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbose": -1,  # Suppress LightGBM's default verbosity
    }

    # 모델 학습
    print("Training LightGBM model...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_train, lgb_eval],
    )
    print("Model training completed.")

    # 예측
    print("예측 진행...")
    y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred = np.round(y_pred_proba)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"테스트셋 정확도: {accuracy:.4f}")
