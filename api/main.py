from fastapi import FastAPI

app = FastAPI()  # FastAPI 인스턴스 생성

@app.get("/")    # GET 요청을 처리하는 엔드포인트
def root():
    return {"message": "Hello~Hi~"}  # JSON 응답 자동 변환