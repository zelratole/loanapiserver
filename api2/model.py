"""
model.py - 대출 심사 ML 모델 래핑 클래스
======================================================
역할: 모델 로딩 + 추론 로직 캡슐화
      schemas.py 의 LoanRequest/LoanResponse 와 연결되는 중간 계층

추론 흐름:
  LoanRequest (영어 dict)
    → FIELD_TO_COLUMN 매핑 (영어 필드명 → 한글 컬럼명)
    → LabelEncoder 인코딩 (범주형 → 숫자)
    → Pipeline.predict_proba (확률 예측)
    → LoanResponse 조립 (결과 dict 반환)
"""

import os
import pickle
import math
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api2.schemas import LoanRequest, LoanResponse

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 필드 매핑: LoanRequest 영어 필드 → 학습 데이터 한글 컬럼명
# ──────────────────────────────────────────────
FIELD_TO_COLUMN: dict[str, str] = {
    "age":              "나이",
    "income":           "연간소득",
    "loan_amount":      "대출금액",
    "loan_term":        "대출기간",
    "interest_rate":    "이자율",
    "credit_score":     "신용점수",
    "employment_type":  "고용형태",
    "education_level":  "교육수준",
    "marital_status":   "결혼여부",
    "loan_purpose":     "대출목적",
    "has_mortgage":     "주택담보대출여부",
    "has_dependents":   "부양가족여부",
}

# 위험 등급 임계값 (승인 확률 기준)
RISK_GRADE_THRESHOLDS = [
    (0.85, "A"),  # 85% 이상 → A등급
    (0.70, "B"),  # 70% 이상 → B등급
    (0.55, "C"),  # 55% 이상 → C등급
    (0.40, "D"),  # 40% 이상 → D등급
    (0.00, "E"),  # 그 외     → E등급
]

APPROVAL_THRESHOLD = 0.5  # 승인/거절 기준 확률


# ──────────────────────────────────────────────
# 모델 래핑 클래스
# ──────────────────────────────────────────────

class LoanPredictor:
    """
    대출 심사 ML 모델 래퍼

    사용법:
        predictor = LoanPredictor()
        predictor.load("models/")
        response = predictor.predict(loan_request)
    """

    def __init__(self) -> None:
        self._pipeline: Any = None          # sklearn Pipeline (전처리 + 분류기)
        self._label_encoders: dict = {}     # {컬럼명: LabelEncoder}
        self._feature_names: list[str] = [] # 학습 시 사용한 컬럼 순서
        self._is_loaded: bool = False

    # ── 모델 로딩 ────────────────────────────────
    def load(self, model_dir: str = "models") -> None:
        """
        모델 아티팩트 3종 로딩
          - loan_pipeline.pkl    : sklearn Pipeline
          - label_encoders.pkl   : 범주형 인코더 dict
          - feature_names.pkl    : 피처 컬럼 순서 list
        """
        model_path = Path(model_dir)

        pipeline_path       = model_path / "loan_pipeline.pkl"
        label_enc_path      = model_path / "label_encoders.pkl"
        feature_names_path  = model_path / "feature_names.pkl"

        # 파일 존재 확인
        for path in [pipeline_path, label_enc_path, feature_names_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"모델 파일을 찾을 수 없습니다: {path}\n"
                    "먼저 train_model.py 를 실행하여 모델을 학습하세요."
                )

        try:
            with open(pipeline_path,      "rb") as f:
                self._pipeline = pickle.load(f)
            with open(label_enc_path,     "rb") as f:
                self._label_encoders = pickle.load(f)
            with open(feature_names_path, "rb") as f:
                self._feature_names = pickle.load(f)

            self._is_loaded = True
            logger.info(
                f"모델 로딩 완료 | 피처 수: {len(self._feature_names)} | "
                f"인코더 컬럼: {list(self._label_encoders.keys())}"
            )

        except Exception as e:
            self._is_loaded = False
            raise RuntimeError(f"모델 로딩 실패: {e}") from e

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ── 핵심 추론 메서드 ──────────────────────────
    def predict(self, request: LoanRequest) -> LoanResponse:
        """
        LoanRequest → LoanResponse 전체 추론 파이프라인

        Step 1. LoanRequest → 한글 컬럼 dict 변환 (FIELD_TO_COLUMN)
        Step 2. bool 필드 → 정수 변환 (True→1, False→0)
        Step 3. 범주형 필드 → LabelEncoder 인코딩
        Step 4. feature_names 순서로 DataFrame 생성
        Step 5. Pipeline.predict_proba 호출
        Step 6. 결과 조립 → LoanResponse 반환
        """
        if not self._is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. load() 를 먼저 호출하세요.")

        # ── Step 1: 영어 필드 → 한글 컬럼 매핑 ──
        raw: dict[str, Any] = request.model_dump()
        mapped: dict[str, Any] = {
            FIELD_TO_COLUMN[field]: value
            for field, value in raw.items()
            if field in FIELD_TO_COLUMN
        }

        # ── Step 2: bool → int 변환, Enum → str(.value) 변환 ──
        for col, val in mapped.items():
            if isinstance(val, bool):
                mapped[col] = int(val)
            elif hasattr(val, "value"):       # Enum 인스턴스 → 한글 문자열
                mapped[col] = val.value

        # ── Step 3: Enum 값(한글 문자열) → LabelEncoder 인코딩 ──
        encoded = dict(mapped)  # 복사
        for col, encoder in self._label_encoders.items():
            if col in encoded:
                str_val = str(encoded[col])
                try:
                    encoded[col] = int(encoder.transform([str_val])[0])
                except ValueError:
                    # 학습 시 없던 범주: 가장 유사한 클래스로 fallback
                    logger.warning(
                        f"[{col}] 미지의 범주 '{str_val}'. 0으로 인코딩합니다."
                    )
                    encoded[col] = 0

        # ── Step 4: feature_names 순서로 DataFrame 구성 ──
        try:
            df = pd.DataFrame([encoded])[self._feature_names]
        except KeyError as e:
            missing = set(self._feature_names) - set(encoded.keys())
            raise ValueError(
                f"입력 데이터에 필요한 피처가 없습니다: {missing}"
            ) from e

        # ── Step 5: 확률 예측 ──
        proba: np.ndarray = self._pipeline.predict_proba(df)
        approval_prob: float = float(proba[0][1])  # 클래스 1 (승인) 확률

        # ── Step 6: 결과 조립 ──
        decision    = "승인" if approval_prob >= APPROVAL_THRESHOLD else "거절"
        risk_grade  = self._get_risk_grade(approval_prob)
        monthly_pay = self._calc_monthly_payment(
            principal     = request.loan_amount,
            annual_rate   = request.interest_rate,
            term_months   = request.loan_term,
        )

        message = (
            f"대출 심사 결과: {decision} "
            f"(승인 확률 {approval_prob * 100:.1f}%, 위험등급 {risk_grade})"
        )

        return LoanResponse(
            approval_probability = round(approval_prob, 4),
            decision             = decision,
            risk_grade           = risk_grade,
            monthly_payment      = round(monthly_pay, 1),
            message              = message,
        )

    # ── 보조 메서드 ───────────────────────────────

    @staticmethod
    def _get_risk_grade(probability: float) -> str:
        """승인 확률 → 위험 등급 (A~E) 변환"""
        for threshold, grade in RISK_GRADE_THRESHOLDS:
            if probability >= threshold:
                return grade
        return "E"

    @staticmethod
    def _calc_monthly_payment(
        principal: float,
        annual_rate: float,
        term_months: int,
    ) -> float:
        """
        원리금 균등상환 월 납부액 계산 (단위: 만원)

        공식: M = P * r(1+r)^n / ((1+r)^n - 1)
          P = 대출원금, r = 월이자율, n = 상환기간(개월)
        """
        if annual_rate == 0:
            return round(principal / term_months, 2)

        monthly_rate = annual_rate / 100 / 12
        n = term_months
        factor = (1 + monthly_rate) ** n
        monthly = principal * monthly_rate * factor / (factor - 1)
        return round(monthly, 2)


# ──────────────────────────────────────────────
# 싱글턴 인스턴스 (main.py 에서 import 하여 사용)
# ──────────────────────────────────────────────
predictor = LoanPredictor()
