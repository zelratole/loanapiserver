"""
schemas.py - 대출 심사 API 입출력 스키마 정의
======================================================
역할: 클라이언트와 서버 간 데이터 계약(Contract) 확정
      스키마가 먼저 확정되어야 모델 래핑과 API 설계가 흔들리지 않음

흐름: LoanRequest → (검증 완료) → LoanResponse
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum


# ──────────────────────────────────────────────
# Enum 정의: 허용 값을 명시적으로 제한
# ──────────────────────────────────────────────

class EducationLevel(str, Enum):
    high_school   = "고등학교"
    college       = "전문대학"
    university    = "대학교"
    graduate      = "대학원"


class EmploymentType(str, Enum):
    full_time   = "정규직"
    contract    = "계약직"
    self_employed = "자영업"
    unemployed  = "무직"


class LoanPurpose(str, Enum):
    home        = "주택구입"
    car         = "자동차"
    education   = "교육"
    business    = "사업"
    personal    = "개인"


class MaritalStatus(str, Enum):
    single   = "미혼"
    married  = "기혼"
    divorced = "이혼"


# ──────────────────────────────────────────────
# 입력 스키마: 클라이언트 → API
# ──────────────────────────────────────────────

class LoanRequest(BaseModel):
    """
    대출 심사 요청 스키마

    필드 이름은 영어(snake_case)로 정의하고,
    model.py 의 FIELD_TO_COLUMN 매핑으로 한글 컬럼명으로 변환됩니다.
    """

    age: int = Field(
        ...,
        ge=18, le=80,
        description="신청자 나이 (18~80세)",
        examples=[35]
    )
    income: float = Field(
        ...,
        ge=0,
        description="연간 소득 (만원)",
        examples=[5000.0]
    )
    loan_amount: float = Field(
        ...,
        gt=0,
        description="대출 희망 금액 (만원)",
        examples=[10000.0]
    )
    loan_term: int = Field(
        ...,
        ge=1, le=360,
        description="대출 기간 (개월, 1~360)",
        examples=[60]
    )
    interest_rate: float = Field(
        ...,
        ge=0.0, le=50.0,
        description="적용 금리 (%, 0~50)",
        examples=[4.5]
    )
    credit_score: int = Field(
        ...,
        ge=300, le=900,
        description="신용점수 (300~900)",
        examples=[720]
    )
    employment_type: EmploymentType = Field(
        ...,
        description="고용 형태",
        examples=["정규직"]
    )
    education_level: EducationLevel = Field(
        ...,
        description="최종 학력",
        examples=["대학교"]
    )
    marital_status: MaritalStatus = Field(
        ...,
        description="결혼 여부",
        examples=["기혼"]
    )
    loan_purpose: LoanPurpose = Field(
        ...,
        description="대출 목적",
        examples=["주택구입"]
    )
    has_mortgage: bool = Field(
        ...,
        description="주택담보대출 보유 여부",
        examples=[False]
    )
    has_dependents: bool = Field(
        ...,
        description="부양가족 보유 여부",
        examples=[True]
    )

    # ── 파생 검증: 소득 대비 대출비율 경고 ──
    @field_validator("loan_amount")
    @classmethod
    def loan_amount_reasonable(cls, v, info):
        """대출 금액이 연 소득의 20배를 초과하면 경고 (통과는 허용)"""
        if "income" in info.data and info.data["income"] > 0:
            ratio = v / info.data["income"]
            if ratio > 20:
                raise ValueError(
                    f"대출금액({v:,.0f}만원)이 연소득({info.data['income']:,.0f}만원)의 "
                    f"{ratio:.1f}배입니다. 최대 20배 이내로 입력해주세요."
                )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "income": 5000.0,
                "loan_amount": 10000.0,
                "loan_term": 60,
                "interest_rate": 4.5,
                "credit_score": 720,
                "employment_type": "정규직",
                "education_level": "대학교",
                "marital_status": "기혼",
                "loan_purpose": "주택구입",
                "has_mortgage": False,
                "has_dependents": True,
            }
        }
    }


# ──────────────────────────────────────────────
# 출력 스키마: API → 클라이언트
# ──────────────────────────────────────────────

class LoanResponse(BaseModel):
    """
    대출 심사 결과 스키마

    approval_probability : 모델 예측 승인 확률 (0.0 ~ 1.0)
    decision             : "승인" | "거절" (threshold=0.5)
    risk_grade           : 위험 등급 (A ~ E)
    monthly_payment      : 예상 월 상환액 (만원)
    message              : 심사 결과 메시지
    """

    approval_probability: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="대출 승인 확률 (0.0 ~ 1.0)",
    )
    decision: Literal["승인", "거절"] = Field(
        ...,
        description="최종 심사 결정",
    )
    risk_grade: Literal["A", "B", "C", "D", "E"] = Field(
        ...,
        description="위험 등급 (A=최우량 ~ E=최고위험)",
    )
    monthly_payment: float = Field(
        ...,
        description="예상 월 상환액 (만원)",
    )
    message: str = Field(
        ...,
        description="심사 결과 안내 메시지",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "approval_probability": 0.823,
                "decision": "승인",
                "risk_grade": "B",
                "monthly_payment": 188.7,
                "message": "대출 심사 결과: 승인 (승인 확률 82.3%, 위험등급 B)",
            }
        }
    }


# ──────────────────────────────────────────────
# 에러 응답 스키마
# ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """표준 에러 응답"""
    detail: str = Field(..., description="에러 상세 메시지")
    code: str   = Field(..., description="에러 코드")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "credit_score 는 300~900 사이여야 합니다.",
                "code": "VALIDATION_ERROR",
            }
        }
    }
