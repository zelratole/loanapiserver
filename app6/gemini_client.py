import json
import logging

from google import genai
from google.genai import types
import os

logger = logging.getLogger(__name__)

REVIEW_SCHEMA = {
            "type" : "object",
            "properties" : {
                "sentiment" : {"type" : "string"},
                "category" : {"type" : "string"},
                "summary" : {"type" : "string"},
                "confidence" : {"type" : "number"}
            },
            "required" : ["sentiment", "category", "summary", "confidence"]
        }

class ReviewAnalyzer:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        #api_key = "AIzaSyAXD7ACCk0oh1-YkST1erwU9AhXChPXfik"
        client = genai.Client(api_key=api_key)


        if not api_key :
            raise ValueError("API 키가 설정되지 않았습니다.")
        self.client = genai.Client(api_key= api_key)
        self.model = "gemini-2.5-flash"
        logger.info("분석기 초기화 완료")

    def analyze(self, review_text:str) -> dict:
        prompt = f"""주어진 리뷰 텍스트를 분석해주세요.

                리뷰 : {review_text}

                다음 기준으로 분석하세요:
                - sentiment : '긍정', '부정', '중립' 중 하나
                - category : '배송', '품질', '가격', '고객서비스', '기타' 중 하나
                - summary : 리뷰 핵심을 1~2문장으로 요약
                - confidence : 0.0 ~ 1.0 사이의 신뢰도
                """
        response = self.client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = prompt,
            config= types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=REVIEW_SCHEMA                
            )
        )

        result = json.loads(response.text)
        return result


