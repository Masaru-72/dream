import os
import uuid
import json
import chromadb
import google.generativeai as genai
import asyncio
from typing import List, Dict

try:
    client = chromadb.PersistentClient(path="chroma_db_store")
    dream_collection = client.get_or_create_collection("dream_journal_ko")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY가 .env 파일에 없습니다.")
    genai.configure(api_key=api_key)
    
    EMBEDDING_MODEL = 'models/text-embedding-004'
    GENERATIVE_MODEL = 'gemini-1.5-flash'

    print("RAG 엔진 초기화 완료 (ChromaDB & Gemini).")

except Exception as e:
    print(f"치명적 오류: RAG 엔진 초기화 중 문제 발생: {e}")
    raise

async def _find_and_add_dream(dream_text: str) -> str:
    """
    이 함수는 RAG의 "검색(Retrieval)" 부분입니다.
    1. 벡터 데이터베이스에서 유사한 꿈을 찾습니다.
    2. 미래의 맥락으로 사용하기 위해 새로운 꿈을 추가합니다.
    가장 유사한 꿈의 텍스트를 반환하거나, 없는 경우 None을 반환합니다.
    """
    try:
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=dream_text,
            task_type="retrieval_query"
        )["embedding"]

        results = dream_collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        
        doc_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=dream_text,
            task_type="retrieval_document"
        )["embedding"]
        
        dream_id = str(uuid.uuid4())
        dream_collection.add(
            documents=[dream_text],
            embeddings=[doc_embedding],
            ids=[dream_id]
        )
        
        if results['documents'] and results['documents'][0] and results['distances'][0][0] < 1.0:
            return results['documents'][0][0]
        
        return None
        
    except Exception as e:
        print(f"ChromaDB 처리 중 오류 발생: {e}")
        return None

async def interpret_dream_rag(dream_text: str, feeling: str, details: List[Dict]) -> str:
    """
    이 함수는 RAG의 핵심인 "생성(Generation)" 부분입니다.
    사용자의 입력을 받아 검색 함수로부터 맥락을 얻고, 완전한 JSON 응답을 생성합니다.
    JSON 문자열을 반환합니다.
    """
    similar_dream = await _find_and_add_dream(dream_text)

    details_str = "\n".join([f"- Q: {d['question']}\n  A: {d['answer']}" for d in details])

    prompt = f"""
    통찰력 있고, 공감 능력이 뛰어나며, 창의적인 꿈 해몽가처럼 행동해주세요.
    사용자가 꿈 내용, 느낀 감정, 그리고 추가 질문에 대한 답변을 제출했습니다.
    당신의 임무는 이 모든 정보를 종합하여 포괄적인 분석을 제공하는 것입니다.
    답변은 반드시 이야기 형식으로 제공해주세요.

    **검색된 맥락 (과거의 비슷한 꿈):**
    "{similar_dream if similar_dream else '기록 없음.'}"
    - 만약 비슷한 꿈이 존재한다면, 해몽 내용에 그 반복되는 패턴이나 주제에 대해 간략하게 언급해주세요.

    **현재 꿈의 세부 정보:**
    - 꿈 내용: "{dream_text}"
    - 사용자가 표현한 감정: "{feeling}"

    **사용자로부터 얻은 추가 정보 (추가 질문):**
    {details_str}
    - 이 답변들을 사용하여 더 구체적이고 미묘한 해몽을 제공해주세요. 예를 들어, 꿈의 배경이 "완전히 낯선 곳"이었다면, 그것이 무엇을 상징하는지 탐구해볼 수 있습니다.

    **응답은 반드시 하나의 유효한 JSON 객체여야 합니다.**
    표시할 때는 사람이 읽기 쉽게 답을 단락으로 나누어야 합니다. 단락의 평균 길이는 4문장이어야 합니다.
    JSON 객체 앞뒤에 어떠한 텍스트, 마크다운, 코드 펜스도 포함하지 마세요.
    JSON 객체는 다음 키들을 포함해야 합니다:
    1. "interpretation": (문자열) 부드러운 대화 톤으로 작성된, 사려 깊은 여러 단락의 해몽. 꿈의 상징을 분석하고 사용자의 답변 및 과거 꿈(발견된 경우)과 연결해주세요. **매우 중요: 해몽 내용에서 가장 핵심적인 키워드나 문구 3개에서 5개를 반드시 HTML 볼드 태그(`<b>`)와 밑줄 태그(`<u>`)를 중첩하여 `<b><u>키워드</u></b>` 형식으로 감싸서 강조해야 합니다. 이것은 필수 사항입니다.** (예: `<b><u>내면의 안정감</u></b>` 또는 `<b><u>새로운 시작</u></b>`).
    2. "themes": (문자열 배열) 사용 가능한 모든 정보에서 파생된 3-5개의 주요 키워드 또는 주제 목록. 각 주제 앞에는 '#'을 붙여 해시태그 형식으로 만들어주세요 (예: `["#자아성찰", "#관계의복잡성"]`).
    3. "image_prompt": (문자열) 꿈의 내용과 사용자의 답변에서 영감을 받아, AI 이미지 생성기가 꿈의 그림을 만들 수 있도록 시각적으로 풍부하고 상세하며 예술적인 프롬프트. **이 프롬프트는 반드시 영어로 작성되어야 합니다.**
    """

    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        response = await model.generate_content_async(
            prompt,
            generation_config=generation_config
        )
        
        if not response.parts:
            print(f"Gemini 응답이 차단되었습니다. 피드백: {response.prompt_feedback}")
            error_payload = {
                "interpretation": "콘텐츠 필터로 인해 요청을 처리할 수 없습니다. 꿈 내용을 다시 작성하여 시도해 주세요.",
                "themes": ["#콘텐츠필터링됨"],
                "image_prompt": "콘텐츠 필터 문제로 프롬프트를 생성할 수 없습니다."
            }
            return json.dumps(error_payload, ensure_ascii=False)
        
        return response.text

    except Exception as e:
        print(f"생성 AI 모델 처리 중 오류 발생: {e}")
        error_payload = {
            "interpretation": "예상치 못한 서버 오류가 발생했습니다. AI 해몽가가 잠시 휴식 중일 수 있습니다. 잠시 후 다시 시도해 주세요.",
            "themes": ["#서버오류"],
            "image_prompt": "서버 오류로 인해 프롬프트를 생성할 수 없습니다."
        }
        return json.dumps(error_payload, ensure_ascii=False)