from contextlib import asynccontextmanager
from http.client import HTTPException
import logging

from app.model import LoanModel
from app.schemas import LoanRequest, LoanResponse
from fastapi import FastAPI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# uvicorn app.main:app --reload  
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('대출 심사 모델을 로드합니다.')
    model = LoanModel()
    try:
        model.load()
        logger.info('모델 로드 성공')
    except Exception as e:
        logger.error(f'모델 로드 실패: {e}')
        logger.warning('/predict 엔드포인트는 모델 로드 후 사용가능')
    app.state.model = model

    yield

    logger.info('대출 심사 API를 종료합니다.')

app = FastAPI(
    title = '대출 심사 예측 API',
    description = 'ML 모델 기반 대출 승인 여부를 예측하는 API',
    version = '1.0.0',
    lifespan = lifespan
)

@app.get('/health')
async def health_check():
    model = app.state.model 
    model_loaded = model.pipeline is not None
    return {
        "status" : "healthy" if model_loaded else "degraded",
        "model_loaded" : model_loaded
    }

@app.post("/predict", response_model = LoanResponse)
async def predict(request: LoanRequest):
    model = app.state.model

    try:

        result = model.predict(request.model_dump())
        LoanResponse(**result)
    
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail="입력값처리오류")
    except Exception as e:
        raise HTTPException(status_code=500)






