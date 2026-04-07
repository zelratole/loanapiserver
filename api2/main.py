"""
main.py - FastAPI 대출 심사 예측 API
======================================================
역할: schemas.py + model.py 를 연결하는 API 진입점
      자동 문서화(/docs), 입력 검증, 에러 처리 통합

엔드포인트:
  POST /predict      - 대출 심사 예측
  GET  /health       - 서버 상태 확인
  GET  /model/info   - 로드된 모델 정보 확인
uvicorn api2.main:app --reload  

uvicorn api2.main:app --reload

"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api2.schemas import LoanRequest, LoanResponse, ErrorResponse
from api2.model import predictor

# ── 로깅 설정 ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 앱 수명주기: 시작 시 모델 로딩 ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드, 종료 시 정리"""
    logger.info("🚀 서버 시작 - 모델 로딩 중...")
    try:
        predictor.load(model_dir="models")
        logger.info("✅ 모델 로딩 완료")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  모델 파일 없음 (데모 모드): {e}")
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
    yield
    logger.info("🛑 서버 종료")


# ── FastAPI 앱 초기화 ─────────────────────────────
app = FastAPI(
    title="대출 심사 예측 API",
    description="""
## 대출 심사 ML 예측 서비스

신청자의 정보를 입력하면 ML 모델이 대출 승인 여부를 예측합니다.

### 처리 흐름
```
클라이언트 요청 → Pydantic 검증 → 모델 추론 → JSON 응답
```

### 위험 등급 기준
| 등급 | 승인 확률   | 의미       |
|------|-----------|------------|
| A    | 85% 이상  | 최우량     |
| B    | 70~84%   | 우량       |
| C    | 55~69%   | 보통       |
| D    | 40~54%   | 주의       |
| E    | 40% 미만  | 고위험     |
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS 미들웨어 ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 운영 환경에서는 도메인 명시
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 처리 시간 미들웨어 ──────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    """응답 헤더에 처리 시간(ms) 추가"""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


# ── 전역 예외 핸들러 ──────────────────────────────
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "code": "VALIDATION_ERROR"},
    )

@app.exception_handler(RuntimeError)
async def runtime_exception_handler(request: Request, exc: RuntimeError):
    logger.error(f"RuntimeError: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "code": "MODEL_ERROR"},
    )


# ══════════════════════════════════════════════
# 라우터 정의
# ══════════════════════════════════════════════

@app.post(
    "/predict",
    response_model=LoanResponse,
    responses={
        200: {"description": "심사 성공", "model": LoanResponse},
        422: {"description": "입력 검증 오류", "model": ErrorResponse},
        503: {"description": "모델 미로드", "model": ErrorResponse},
    },
    summary="대출 심사 예측",
    tags=["예측"],
)
async def predict_loan(request: LoanRequest) -> LoanResponse:
    """
    ## 대출 심사 예측

    신청자 정보를 입력하면 ML 모델이 승인 확률과 위험 등급을 반환합니다.

    ### 입력 필드
    - **age**: 나이 (18~80세)
    - **income**: 연간 소득 (만원)
    - **loan_amount**: 대출 희망 금액 (만원)
    - **loan_term**: 대출 기간 (개월)
    - **interest_rate**: 적용 금리 (%)
    - **credit_score**: 신용점수 (300~900)
    - **employment_type**: 고용 형태 (정규직/계약직/자영업/무직)
    - **education_level**: 최종 학력 (고등학교/전문대학/대학교/대학원)
    - **marital_status**: 결혼 여부 (미혼/기혼/이혼)
    - **loan_purpose**: 대출 목적 (주택구입/자동차/교육/사업/개인)
    - **has_mortgage**: 주택담보대출 보유 여부
    - **has_dependents**: 부양가족 보유 여부
    """
    # 모델 로드 확인
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="모델이 로드되지 않았습니다. 서버 관리자에게 문의하세요.",
        )

    # 추론 실행
    logger.info(
        f"심사 요청 | 나이={request.age} | 소득={request.income:,.0f}만원 "
        f"| 대출={request.loan_amount:,.0f}만원 | 신용점수={request.credit_score}"
    )

    try:
        result = predictor.predict(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("예측 중 예상치 못한 오류 발생")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"예측 오류: {e}",
        )

    logger.info(
        f"심사 결과 | 결정={result.decision} "
        f"| 확률={result.approval_probability:.1%} | 등급={result.risk_grade}"
    )
    return result


@app.get(
    "/health",
    summary="헬스 체크",
    tags=["운영"],
)
async def health_check():
    """서버 및 모델 상태 확인"""
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded,
    }


@app.get(
    "/model/info",
    summary="모델 정보 조회",
    tags=["운영"],
)
async def model_info():
    """로드된 모델의 피처 수, 인코더 정보 반환"""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="모델이 로드되지 않았습니다.",
        )
    return {
        "feature_count":    len(predictor._feature_names),
        "feature_names":    predictor._feature_names,
        "encoder_columns":  list(predictor._label_encoders.keys()),
        "approval_threshold": 0.5,
        "risk_grades": {
            "A": "승인 확률 85% 이상",
            "B": "승인 확률 70~84%",
            "C": "승인 확률 55~69%",
            "D": "승인 확률 40~54%",
            "E": "승인 확률 40% 미만",
        },
    }


# ── 개발 서버 직접 실행 ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # 개발 시 코드 변경 자동 반영
        log_level="info",
    )
