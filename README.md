# 대출 심사 예측 API (Loan Approval Prediction API)

대출 신청자의 신원정보를 입력받아 대출 승인 여부를 예측하는 FastAPI 기반 머신러닝 서비스입니다.

## 🎯 주요 기능

- **대출 심사 예측**: 신청자 정보로부터 대출 승인/거절 예측
- **자동 문서화**: Swagger UI (`/docs`)를 통한 API 문서 제공
- **입력 검증**: Pydantic을 이용한 자동 데이터 검증
- **에러 처리**: 구조화된 에러 응답
- **모델 정보 조회**: 현재 로드된 모델 정보 확인

## 📋 필수 요구사항

- Python 3.8+
- pip 또는 conda

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/loanapiserver.git
cd loanapiserver
```

### 2. 가상 환경 생성
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 설정
`.env` 파일을 생성하고 필요한 環경 변수 설정:
```
GEMINI_API_KEY=your_api_key_here
```

### 5. 개발 서버 실행
```bash
uvicorn api2.main:app --reload
```

서버가 시작되면 다음 주소에서 API에 접근할 수 있습니다:
- **API 문서**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API 엔드포인트

### 대출 심사 예측
```http
POST /predict
```

**요청 예시:**
```json
{
  "age": 35,
  "income": 50000,
  "credit_score": 720,
  "loan_amount": 100000,
  "employment_years": 5
}
```

**응답:**
```json
{
  "prediction": "approved",
  "confidence": 0.85,
  "message": "대출이 승인되었습니다."
}
```

### 서버 상태 확인
```http
GET /health
```

### 모델 정보 조회
```http
GET /model/info
```

## 📁 프로젝트 구조

```
loanapiserver/
├── api/                 # 기본 구현
├── api2/                # 고급 구현 (권장)
│   ├── main.py         # API 진입점
│   ├── model.py        # 모델 로직
│   ├── schemas.py      # 요청/응답 스키마
│   ├── train_model.py  # 모델 학습 스크립트
│   └── requirements.txt # 의존성
├── app/                 # 대체 구현
├── models/              # 학습된 모델 저장소
├── .env                 # 환경 설정 (git에서 제외)
├── .gitignore          # git 무시 파일
├── requirements.txt     # 프로젝트 의존성
└── README.md           # 이 파일
```

## 🔧 모델 학습

다음 명령어로 새로운 모델을 학습할 수 있습니다:
```bash
python api2/train_model.py
```

## 📦 의존성

- **fastapi** >= 0.115.0 - 웹 API 프레임워크
- **uvicorn** >= 0.30.0 - ASGI 웹 서버
- **pydantic** >= 2.7.0 - 데이터 검증
- **scikit-learn** >= 1.4.0 - 머신러닝 라이브러리
- **pandas** >= 2.2.0 - 데이터 처리
- **numpy** >= 1.26.0 - 수치 연산

자세한 내용은 [requirements.txt](requirements.txt)를 참조하세요.

## 🧪 테스트

```bash
pytest tests/
```

## 📝 라이선스

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.

## 👥 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 📧 연락처

질문 사항은 이슈를 등록해주세요.
