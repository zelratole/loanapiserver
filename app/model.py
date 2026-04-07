import os
import logging
import joblib
import pandas as pd 
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

FIELD_TO_COLUMN = {
    "age": "나이",
    "gender": "성별",
    "annual_income": "연소득",
    "employment_years": "근속연수",
    "housing_type": "주거형태",
    "credit_score": "신용점수",
    "existing_loan_count": "기존대출건수",
    "annual_card_usage": "연간카드사용액",
    "debt_ratio": "부채비율",
    "loan_amount": "대출신청액",
    "loan_purpose": "대출목적",
    "repayment_method": "상환방식",
    "loan_period": "대출기간",
}

class LoanModel :
    def __init__(self):
        self.pipeline = None
        self.label_encoders: dict[str, Any] = {}
        self.feature_names: list[str] = []
        self.threshold: float = 0.5
        self.model_version: str = "1.0.0"

    def load(self, model_dir: str = "models") -> None:
        # 스크립트 위치를 기준으로 모델 디렉토리 경로 계산
        script_dir = Path(__file__).parent
        model_path = script_dir.parent / model_dir
        
        pipeline_path = model_path / "loan_pipeline.pkl"
        encoder_path = model_path / "label_encoders.pkl"
        feature_names_path = model_path / "feature_names.pkl"

        if not pipeline_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {pipeline_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"라벨 인코더 파일을 찾을 수 없습니다: {encoder_path}")
        if not feature_names_path.exists():
            raise FileNotFoundError(f"특성 이름 파일을 찾을 수 없습니다: {feature_names_path}")

        self.pipeline = joblib.load(pipeline_path)
        self.label_encoders = joblib.load(encoder_path)
        self.feature_names = joblib.load(feature_names_path)

        logging.info("모델 로드 완료")
    
    @staticmethod
    def _map_to_korean(data: dict[str, Any]) -> dict[str, Any]:
        return {FIELD_TO_COLUMN.get(k, k): v for k, v in data.items()}
    
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load()함수를 먼저 호출하세요.")
        
        mapped = self._map_to_korean(data)
        df = pd.DataFrame( [ mapped ] )[self.feature_names]

        for col, encoder in self.label_encoders.items() :
            df[col] = encoder.transform( df[col] )

        probability = float( self.pipeline.predict_proba( df )[0,1] )
        approved = probability >= self.threshold
        risk_grade = self._get_risk_grade(probability)

        return {
            "approved" : approved,
            "probability" : probability,
            "risk_grade" : risk_grade
        }


    @staticmethod
    def _get_risk_grade(probalility: float) -> str:
        if probalility >= 0.75:
            return "A"
        elif probalility >= 0.5 :
            return "B"
        elif probalility >= 0.25 :
            return "C"
        else :
            return "D"