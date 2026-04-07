import os
import logging
import joblib
import pandas as pd 
from typing import Any

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
        pipeline_path = os.path.join( model_dir,"loan_pipeline.pkl")
        encoder_path = os.path.join(model_dir, "label_encoders.pkl")
        feature_names_path=os.path.join(model_dir, "feature_names.pkl")

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