from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length = 1,
        max_length = 5000, 
        description = "분석할 고객 리뷰 텍스트",
        examples=["이 제품 정말 좋아요! 배송도 빠르고 품질이 우수합니다."]
    )

class ReviewResponse(BaseModel):
    sentiment: str = Field(
        ...,
        description= "감성 분석 결과",
        examples=["긍정", "부정", "중립"]
    )

    category: str = Field(
        ...,
        description="리뷰 카테고리",
        examples=["품질", "배송", "가격", "서비스", "기타"]
    )

    summary: str = Field(
        ...,
        description="리뷰 요약 (1~2문장)"
    )

    confidence: float = Field(
        ...,
        ge = 0.0,
        le = 1.0,
        description="분석 신뢰도 (0.0~1.0)"
    )