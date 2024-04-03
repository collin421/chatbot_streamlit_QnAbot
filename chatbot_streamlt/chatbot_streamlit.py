# 스트림릿 임포트
import streamlit as st
from streamlit_option_menu import option_menu
 
import tiktoken

# 기록 불러 오기
from loguru import logger

from langchain.chains import ConversationalRetrievalChain

# llm 모델 불러오기
from langchain.chat_models import ChatOpenAI

# 랭체인 로더 불러 오기
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import WebBaseLoader

# 텍스트 스플릿터 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 임베딩 모델 불러오기
from langchain.embeddings import HuggingFaceEmbeddings

# 채팅 버퍼 메모리
from langchain.memory import ConversationBufferMemory

# 벡터스토어 불러오기(메모리 사용 faiss)
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from dotenv import load_dotenv
import os

import sqlite3
from sqlite3 import Error

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="DirChat",page_icon=":books:")

def main_page():
    st.title("Main Page")
    st.write("안녕하세요! 이것은 메인 페이지입니다.")

# SQLite 데이터베이스 연결 함수
def create_connection(db_file):
    """
    SQLite 데이터베이스 파일에 연결합니다.
    연결 실패 시 None을 반환합니다.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        st.error(f"데이터베이스 연결 에러: {e}")
    return conn

# SQLite 테이블 생성 함수
def create_table(conn, create_table_sql):
    """
    주어진 SQL 명령문(create_table_sql)을 사용하여 SQLite 데이터베이스에 테이블을 생성합니다.
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        st.error(f"테이블 생성 에러: {e}")

# 사용자 추가 함수
def create_user(conn, user):
    """
    'users' 테이블에 새로운 사용자를 추가합니다.
    사용자(username, password) 정보는 튜플 형태로 제공됩니다.
    """
    sql = ''' INSERT INTO users(username, password) VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()  # 데이터베이스에 변경사항을 반영합니다.
    return cur.lastrowid  # 생성된 사용자의 ID를 반환합니다.

# 사용자 존재 여부 확인 함수
def check_user_exists(conn, username):
    """
    주어진 사용자 이름으로 'users' 테이블을 조회하여 사용자가 존재하는지 확인합니다.
    사용자가 존재하면 True, 그렇지 않으면 False를 반환합니다.
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    rows = cur.fetchall()
    return len(rows) > 0

# 회원 가입 페이지 구현
def signup_page():
    """Streamlit을 사용하여 회원 가입 페이지를 구현합니다."""
    st.title("회원 가입")

    # 데이터베이스 파일명 정의
    db_file = "database.db"
    
    # 데이터베이스 연결 및 'users' 테이블 생성
    conn = create_connection(db_file)
    if conn is not None:
        create_table_sql = """ CREATE TABLE IF NOT EXISTS users (
                                            id INTEGER PRIMARY KEY,
                                            username TEXT NOT NULL,
                                            password TEXT NOT NULL
                                        ); """
        create_table(conn, create_table_sql)
    else:
        st.error("데이터베이스 연결에 실패했습니다.")
    
    # 회원 가입 폼 생성
    with st.form("Signup Form"):
        username = st.text_input("사용자 이름", help="회원 가입할 사용자 이름을 입력하세요.")
        password = st.text_input("비밀번호", type="password", help="비밀번호를 입력하세요.")
        submit_button = st.form_submit_button("회원 가입")
        
        # 폼 제출 처리
        if submit_button:
            # 사용자 이름이 이미 존재하는지 확인
            if check_user_exists(conn, username):
                st.error("이미 존재하는 사용자 이름입니다.")
            else:
                # 새 사용자 정보를 데이터베이스에 추가
                create_user(conn, (username, password))
                st.success("회원 가입에 성공했습니다!")

def chatbot_page():

    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        process = st.button("Process")
    if process:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "파일을 업로드하고 나만의 챗봇을"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain
# 멀티페이지 설정
with st.sidebar:
    selected = option_menu("Main Menu", ["Main", "Signup", "Chatbot"], icons=["house", "robot", "chat"], menu_icon="cast", default_index=0)
    
# 페이지별 내용을 렌더링
if selected == "Main":
    main_page()

elif selected == "Signup":
    signup_page()

elif selected == "Chatbot":
    chatbot_page()