import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 🔥 핵심 함수
def get_ai_message(user_message):

    # 1. Embedding + DB
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "tax-table-index"

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )

    retriever = database.as_retriever(search_kwargs={"k": 4})

    # 2. LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 3. dictionary 체인 (질문 변환)
    dictionary_prompt = ChatPromptTemplate.from_template("""
사용자의 질문을 세법 기준으로 더 명확하게 변환하세요.

[규칙]
- '직장인', '회사원', '일반인', '근로자' → '거주자'로 변환
- 의미 변경 금지
- 변환된 질문만 출력

질문:
{question}
""")

    dictionary_chain = (
        dictionary_prompt
        | llm
        | StrOutputParser()
    )

    # 4. RAG Prompt
    rag_prompt = ChatPromptTemplate.from_template("""
당신은 대한민국 소득세 계산 전문가입니다.

아래 문맥을 기반으로 질문에 답하세요.

[규칙]
- 반드시 대한민국 세법 기준
- 계산 과정 포함
- 모르면 모른다고 답변

질문:
{question}

문맥:
{context}

답변:
""")

    # 5. RAG 체인
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 🔥 최종 체인
    final_chain = dictionary_chain | rag_chain

    return final_chain.invoke(user_message)


# 🔥 UI
if user_question := st.chat_input("소득세 관련 질문을 입력하세요"):

    with st.chat_message("user"):
        st.write(user_question)

    st.session_state.message_list.append({
        "role": "user",
        "content": user_question
    })

    with st.spinner("답변을 생성하고 있어용"):
        with st.chat_message("ai"):
            ai_message = get_ai_message(user_question)
            st.write(ai_message)

        st.session_state.message_list.append({
            "role": "ai",
            "content": ai_message
        })