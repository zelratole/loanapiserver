# app/main.py (최종본)
'''
uvicorn app5.main:app --reload로 실행 후, 
http://localhost:8000/health
http://localhost:8000/predict


'''
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException # 수정: fastapi에서 HTTPException 임포트
import logging

from app.model import LoanModel
from app.schemas import LoanRequest, LoanResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('대출 심사 모델을 로드합니다.')
    model = LoanModel()
    try:
        model.load() # 5단계에서 구현할 실제 로드 함수 호출
        logger.info('모델 로드 성공')
    except Exception as e:
        logger.error(f'모델 로드 실패: {e}')
        logger.warning('/predict 엔드포인트는 모델 로드 후 사용가능')
    
    app.state.model = model # 상태 객체에 모델 저장
    yield
    logger.info('대출 심사 API를 종료합니다.')

app = FastAPI(
    title='대출 심사 예측 API (4단계)',
    description='ML 모델 기반 대출 승인 여부를 예측하는 API',
    version='1.0.0',
    lifespan=lifespan
)

@app.get('/health')
async def health_check():
    model = app.state.model 
    model_loaded = model.pipeline is not None
    return {
        "status" : "healthy" if model_loaded else "degraded",
        "model_loaded" : model_loaded
    }

@app.post("/predict", response_model=LoanResponse)
async def predict(request: LoanRequest):
    model = app.state.model

    try:
        result = model.predict(request.model_dump())
        return LoanResponse(**result) # 수정: 클라이언트에게 결과를 주기 위해 return 추가
    
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail="입력값처리오류")
    except Exception as e:
        logger.error(f"예측 중 알 수 없는 에러 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 에러")


