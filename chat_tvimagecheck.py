import streamlit as st
import os
from pathlib import Path
import base64
import json
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(page_title="TV 이미지 판독기", page_icon="😎")

st.title("🤖 TV 이미지 판독기")
st.caption("폴더 내 이미지 자동 분석")

# -------------------------
# 🔥 LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------
# 🔥 이미지 변환
# -------------------------
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -------------------------
# 🔥 JSON 파싱
# -------------------------
def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        return json.loads(text[start:end+1])

    raise ValueError("JSON 파싱 실패")

# -------------------------
# 🔥 이미지 분석
# -------------------------
def analyze_image(image_path):

    base64_img = image_to_base64(image_path)

    prompt = f"""
당신은 TV 화면 불량 분석 AI입니다.

카테고리 중 하나 선택:
- 정상
- 가로줄
- 세로줄
- 점불량
- 얼룩/번짐
- 밝기불균일
- 복합불량

JSON만 출력:
{{
  "category": "",
  "confidence": 0.0,
  "summary": ""
}}
"""

    message = HumanMessage(
    content=[
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        }
    ]
)

    response = llm.invoke([message])
    content = response.content

    if isinstance(content, list):
        content = "".join([c.get("text", "") for c in content if isinstance(c, dict)])

    return extract_json(content)

# -------------------------
# 🔥 배치 처리
# -------------------------
def run_batch(folder):

    rows = []
    folder_path = Path(folder)

    if not folder_path.exists():
        raise ValueError("폴더가 존재하지 않습니다")

    files = list(folder_path.glob("*"))

    for path in files:
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            r = analyze_image(str(path))

            rows.append({
                "file": path.name,
                "category": r["category"],
                "confidence": r["confidence"],
                "summary": r["summary"]
            })

        except Exception as e:
            rows.append({
                "file": path.name,
                "category": "ERROR",
                "confidence": None,
                "summary": str(e)
            })

    df = pd.DataFrame(rows)

    output_path = folder_path / "result.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df, output_path

# -------------------------
# 🔥 UI
# -------------------------

folder_path = st.text_input("📁 이미지 폴더 경로 입력", "")

if st.button("🚀 검토 시작"):

    if not folder_path:
        st.warning("폴더 경로를 입력하세요")
    else:
        with st.spinner("이미지 분석 중..."):
            try:
                df, output_file = run_batch(folder_path)

                st.success("분석 완료!")

                st.dataframe(df)

                with open(output_file, "rb") as f:
                    st.download_button(
                        label="📥 결과 다운로드",
                        data=f,
                        file_name="result.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(str(e))