# app/main.py
'''
uvicorn app10.main:app --reload로 실행 후, 
http://localhost:8000/
http://localhost:8000/generate


pip install -U google-genai


uvicorn main:app --reload

http://localhost:8000/docs





'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

# 1. FastAPI 애플리케이션 초기화
app = FastAPI(    title="고객 리뷰 분석 API(변경)",
    description="Gemini LLM 기반 고객(변경)",
    version="1.0.2")

# 2. Gemini 클라이언트 초기화 (발급받으신 API 키 적용)
# 주의: 실제 서비스 배포 시에는 API 키를 코드에 직접 쓰지 않고 환경 변수로 관리해야 합니다.
# 
API_KEY = "AIzaSyAXD7ACCk0oh1-YkST1erwU9AhXChPXfik"
client = genai.Client(api_key=API_KEY)

# 3. 클라이언트로부터 받을 요청 데이터 형식 정의
class PromptRequest(BaseModel):
    prompt: str

# 4. 텍스트 생성을 위한 API 엔드포인트 (POST 방식)
@app.post("/generate")
async def generate_text(request: PromptRequest):
    try:
        # 최신 Gemini 2.5 Flash 모델을 사용하여 텍스트 생성
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.prompt,
        )
        # 정상적으로 생성된 텍스트 반환
        return {"response": response.text}
    
    except Exception as e:
        # 에러 발생 시 상태 코드 500과 에러 내용 반환
        raise HTTPException(status_code=500, detail=str(e))

# 5. 서버 상태 확인용 기본 엔드포인트
@app.get("/")
async def root():
    return {"message": "Gemini API 서버가 정상적으로 실행 중입니다."}
