# app/main.py
'''
uvicorn app2.main:app --reload로 실행 후, 
http://localhost:8000/health
'''
from fastapi import FastAPI

app = FastAPI(
    title='대출 심사 예측 API (1단계)',
    description='가장 기본적인 API 뼈대',
    version='1.0.0'
)

@app.get('/health')
async def health_check():
    # 모델이 아직 없으므로 단순하게 정상(healthy) 상태만 반환합니다.
    return {
        "status": "healthy",
        "model_loaded": False
    }