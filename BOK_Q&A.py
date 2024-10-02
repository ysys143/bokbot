#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
# 사전정의된 Query-Answer가 담긴 테이블
qna_df = pd.read_csv('qa_data.csv')[['질문', '답변']]

qna_df['질문'] = qna_df['질문'].apply(lambda x: x.split('질문\n')[1]) # "질문\n" 제거
qna_df['답변'] = qna_df['답변'].apply(lambda x: x.split('답변\n')[1]) # "답변\n" 제거


# In[10]:


from sentence_transformers import SentenceTransformer

# SentenceTransformer 모델 로드
embedding_model = SentenceTransformer('jeonseonjin/embedding_BAAI-bge-m3')

# 쿼리 문장들에 대한 임베딩 벡터 생성
query_texts = qna_df['질문'].to_list()
query_embeddings = embedding_model.encode(query_texts)


# In[11]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# query-answer 함수 정의
def qna_answer_to_query(new_query, embedding_model=embedding_model, query_embeddings=query_embeddings, top_k=1, verbose=True):
    # 쿼리 임베딩 계산
    new_query_embedding = embedding_model.encode([new_query])

    
    # 코사인 유사도 계산
    cos_sim = cosine_similarity(new_query_embedding, query_embeddings)
    
    # 코사인 유사도 값이 가장 큰 질문의 인덱스 찾기
    most_similar_idx = np.argmax(cos_sim)
    similarity = np.round(cos_sim[0][most_similar_idx], 2)
    
    # 가장 비슷한 질문과 답변 가져오기
    similar_query = query_texts[most_similar_idx]
    similar_answer = qna_df.iloc[most_similar_idx]['답변']
    
    if verbose == True:
        print("가장 비슷한 질문 : ", similar_query)
        print("가장 비슷한 질문의 유사도 : ", similarity)
        print("가장 비슷한 질문의 답: ", similar_answer)

    # 결과 반환
    return similar_query, similarity, similar_answer


# In[12]:


qna_answer_to_query('전세계 외환시장의 거래규모')


# In[14]:


import gradio as gr

# 질문에 대한 답변을 제공하는 함수 (qna_answer_to_query 함수 사용)
def chat_with(message, history):
    # 사용자의 질문에 대해 full_answer_to_query를 사용하여 답변 생성
    response = qna_answer_to_query(message)[2]
    
    # 질문과 답변을 히스토리에 저장 (history는 대화 히스토리)
    history.append((message, response))  
    
    # Gradio가 (응답, history)를 반환해야 하므로, 대화 기록과 함께 반환
    return history, history


# Gradio Chatbot 인터페이스 생성
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()  # 대화 기록을 표시하는 컴포넌트
    msg = gr.Textbox(label="질문 입력")  # 질문 입력을 위한 텍스트 박스
    clear = gr.Button("대화 기록 초기화")  # 대화 기록 초기화 버튼

    # 대화가 시작될 때 실행할 동작 정의
    msg.submit(chat_with, inputs=[msg, chatbot], outputs=[chatbot, msg])  # 입력값을 처리 후 출력

    # 기록 초기화 버튼 동작 정의
    clear.click(lambda: [], None, chatbot, queue=False)  # 대화 기록을 초기화

# 앱 실행
demo.launch(share=True)

