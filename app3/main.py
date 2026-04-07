# app/main.py
'''
uvicorn app3.main:app --reload로 실행 후, 
http://localhost:8000/health
http://localhost:8000/predict
'''
# app/main.py (업데이트)
from fastapi import FastAPI
from app.schemas import LoanRequest, LoanResponse # 스키마 추가

app = FastAPI(title='대출 심사 예측 API (2단계)', version='1.0.0')

@app.get('/health')
async def health_check():
    return {"status": "healthy", "model_loaded": False}

@app.post("/predict", response_model=LoanResponse)
async def predict(request: LoanRequest):
    # 2단계: 실제 모델이 없으므로, 데이터가 잘 들어왔다고 가정하고 임시 결과를 줍니다.
    return LoanResponse(
        approved=True,
        probability=0.85,
        risk_grade="A"
    )