"""
train_model.py - 대출 심사 모델 학습 및 아티팩트 저장
======================================================
실행:  python train_model.py

생성 파일:
  models/loan_pipeline.pkl    - sklearn Pipeline (전처리 + RandomForest)
  models/label_encoders.pkl   - 범주형 컬럼 인코더 dict
  models/feature_names.pkl    - 피처 컬럼 순서 list
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ── 재현성 고정 ───────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── 범주형 컬럼 및 허용값 정의 ───────────────────────
CATEGORICAL_COLS = {
    "고용형태":   ["정규직", "계약직", "자영업", "무직"],
    "교육수준":   ["고등학교", "전문대학", "대학교", "대학원"],
    "결혼여부":   ["미혼", "기혼", "이혼"],
    "대출목적":   ["주택구입", "자동차", "교육", "사업", "개인"],
}

NUMERIC_COLS = ["나이", "연간소득", "대출금액", "대출기간", "이자율", "신용점수"]
BOOL_COLS    = ["주택담보대출여부", "부양가족여부"]


def generate_dataset(n: int = 2000) -> pd.DataFrame:
    """합성 대출 데이터셋 생성"""
    df = pd.DataFrame({
        "나이":       np.random.randint(20, 70, n),
        "연간소득":   np.random.exponential(4000, n).clip(500, 20000),
        "대출금액":   np.random.exponential(8000, n).clip(500, 50000),
        "대출기간":   np.random.choice([12, 24, 36, 60, 120, 240], n),
        "이자율":     np.random.uniform(2.0, 15.0, n).round(2),
        "신용점수":   np.random.randint(300, 900, n),
        "고용형태":   np.random.choice(CATEGORICAL_COLS["고용형태"], n,
                         p=[0.5, 0.25, 0.15, 0.10]),
        "교육수준":   np.random.choice(CATEGORICAL_COLS["교육수준"], n,
                         p=[0.20, 0.20, 0.45, 0.15]),
        "결혼여부":   np.random.choice(CATEGORICAL_COLS["결혼여부"], n,
                         p=[0.35, 0.55, 0.10]),
        "대출목적":   np.random.choice(CATEGORICAL_COLS["대출목적"], n),
        "주택담보대출여부": np.random.randint(0, 2, n),
        "부양가족여부":    np.random.randint(0, 2, n),
    })

    # 승인 여부 (비즈니스 규칙 + 노이즈)
    score = (
          (df["신용점수"] - 300) / 600 * 0.35
        + (df["연간소득"] / 20000)      * 0.25
        + (1 - df["대출금액"] / 50000)  * 0.20
        + (df["고용형태"] == "정규직").astype(float) * 0.10
        + (df["나이"].between(30, 55)).astype(float)  * 0.10
        + np.random.normal(0, 0.05, n)
    ).clip(0, 1)

    df["대출승인"] = (score >= 0.50).astype(int)
    return df


def train_and_save():
    print("=" * 55)
    print("  대출 심사 모델 학습 시작")
    print("=" * 55)

    # 1. 데이터 생성
    df = generate_dataset(2000)
    print(f"\n[1] 데이터셋 생성: {len(df):,}건  "
          f"(승인 {df['대출승인'].mean():.1%})")

    # 2. 범주형 인코딩
    label_encoders: dict[str, LabelEncoder] = {}
    for col, values in CATEGORICAL_COLS.items():
        le = LabelEncoder()
        le.fit(values)
        df[col] = le.transform(df[col])
        label_encoders[col] = le

    print(f"[2] LabelEncoder 학습 완료: {list(label_encoders.keys())}")

    # 3. 피처 / 타겟 분리
    feature_names = NUMERIC_COLS + list(CATEGORICAL_COLS.keys()) + BOOL_COLS
    X = df[feature_names]
    y = df["대출승인"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[3] 학습 {len(X_train):,}건 / 검증 {len(X_test):,}건")

    # 4. 파이프라인 구성 및 학습
    pipeline = Pipeline([
        ("scaler",      StandardScaler()),
        ("classifier",  RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("[4] 파이프라인 학습 완료")

    # 5. 성능 평가
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n[5] 모델 성능 (검증셋)")
    print(f"    AUC-ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["거절", "승인"]))

    # 6. 아티팩트 저장
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    with open(model_dir / "loan_pipeline.pkl",   "wb") as f:
        pickle.dump(pipeline, f)
    with open(model_dir / "label_encoders.pkl",  "wb") as f:
        pickle.dump(label_encoders, f)
    with open(model_dir / "feature_names.pkl",   "wb") as f:
        pickle.dump(feature_names, f)

    print(f"[6] 모델 아티팩트 저장 완료 → {model_dir.resolve()}/")
    print("\n✅ 학습 완료! 이제 서버를 시작할 수 있습니다.")
    print("   uvicorn app.main:app --reload")
    print("=" * 55)


if __name__ == "__main__":
    train_and_save()
