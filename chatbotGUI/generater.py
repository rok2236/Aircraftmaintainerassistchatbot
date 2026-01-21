#  모델을 불러와 답을 생성하기 위한 생성기 코드
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from chatbotGUI.retriever import rerank_retriever
from chatbotGUI.models_loader import LLM_REWRITE, SAMPLING_PARAMS_REWRITE, TOKENIZER_REWRITE, LLM_MAIN, SAMPLING_PARAMS_MAIN
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
import csv

import tensorflow.compat.v1 as tf1
import os

#nvidia-smi 명령어를 터미널에서 사용하여 사용가능한 GPU의 넘버를 확인할 것
# GPU 번호 지정 '0'은 1번 GPU,'1'은 2번 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
config = tf1.ConfigProto()

# GPU 사용량 지정 0.03은 3%를 사용하겠다는 뜻
# 가능한 3% 이하로 사용할 것
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf1.Session(config=config)



def models_run(querylist):
    results = {}
    queries = []
    for q in querylist:
        messages = [
                {"role": "system",
                 "content": """You are an AI designed to refine and clarify user questions for an aircraft maintenance specialist model. Your task is to:
                1. Correct any typos or grammatical errors while maintaining the technical terminology.
                2. Reorganize the questions to ensure they are concise and contextually clear.
                3. Maintain a professional tone suitable for technical documentation.
                4. Do not provide any answers, solutions, or explanations to the underlying question. Your sole purpose is to produce an improved version of the user's question.
                If the user asks for clarification on the APU, engine operation, or any other technical topic, you must refrain from answering directly and instead return a refined question.
                5. Even if the user asks unrelated questions (e.g., greetings like "Hello" or "What is your role?"), do not provide any information or explanations. Instead, refine and restate the user's question in a clearer and more concise manner without addressing the content. Your sole role is to improve the question itself, not to answer it.
                """},
                {"role": "user", "content": "PCS의 작동 원리중 디지털 전자 제어 (DEC)에서 앤진의 출력 터빈 속도 (Np)는 어덯게 관리하나료"},
                {"role": "assistant", "content": "PCS의 디지털 전자 제어(DEC)가 출력 터빈 속도(Np)를 제어하는 방식은 무엇인가요?"},
                {"role": "user", "content": "FORWARD SUPORT TUBE / ENGINE OUTPUT SHAFT 재거 방법이 어디에 있는지 알랴줘"},
                {"role": "assistant", "content": "FORWARD SUPPORT TUBE와 ENGINE OUTPUT SHAFT를 제거하는 방법은 무엇인가요?"},
                {"role": "user", "content": "Engine Indicating Sytem에는 어떤것이 있눈지 알려줘"},
                {"role": "assistant", "content": "Engine Indicating System에는 어떤 구성 요소가 포함되어 있나요?"},
                {"role": "user", "content": "APU 이가 무앗이야?"},
                {"role": "assistant", "content": "APU는 무엇인가요?"},
                {"role": "user", "content": "안녕!"},
                {"role": "assistant", "content": "안녕하세요."},
                {"role": "user", "content": "네 역할이 뭔지 알려줘."},
                {"role": "assistant", "content": "당신의 역할은 무엇인가요?"},
                {"role": "user", "content": q}
            ]
        
        text =  TOKENIZER_REWRITE.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True
        )
                
        outputs = LLM_REWRITE.generate(text, sampling_params=SAMPLING_PARAMS_REWRITE)

        for output in outputs:
            prompt = output.prompt
            generated_rewrite_text = output.outputs[0].text
        
#         rerank---------------------------------

        format_docs, metadata = rerank_retriever(generated_rewrite_text)

        content = []
        for i in range(len(format_docs)):
            content.append(format_docs[i][1])

        pattern = r"Context:\s*(.*)"
        contents = [re.search(pattern, text, re.DOTALL).group(1) for text in content]

        pattern = r"(.*)Context:"
        pages = [re.search(pattern, text, re.DOTALL).group(1) for text in content]
        
        unique_pages = ", ".join(set(pages))
        doc_path = f"해당 답과 관련있는 페이지는 {unique_pages}입니다."

        SYSTEM_PROMPT  = f"""
        You are an intelligent assistant helping the users with their questions on Aircraft Mechanic. Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

        Do not try to make up an answer:
        - If the answer to the question cannot be determined from the context alone, say "Due to limited information, I am unable to offer a precise solution."
        - If the context is empty, just say "I do not know the answer to that."
        Instructions:
        - Answer the query using only the provided context.
        - All answers should be translated into Korean.
        - Please provide a general answer if the content related to the question is not found in the context.
        - If asked about an abbreviation, the answer must include its full form and an explanation.
        - Do not generate repetitive sentences in your response.
        - If a question unrelated to aircraft maintenance is asked, please disregard the '# Context' sections and provide a response to the question.
        - If '# Context' includes numerical information, you must only use the numerical information provided within that context.

        # Context:
        {contents}
        # Write a passage that answers the given query:
        Question:
        {generated_rewrite_text}
        # Answer
        Passage:
        """   
        # ===  models  ===
        prompt = [SYSTEM_PROMPT]
        outputs = LLM_MAIN.generate(prompt, SAMPLING_PARAMS_MAIN)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_model_text = output.outputs[0].text
            final_answer = f'''{generated_model_text}\n\n{doc_path}'''

            print(f"Question: {q!r}")
            print('-----------------------')
            print(f"0. Rewrite Prompt: {messages!r}")
            print('-----------------------')
            print(f"1. Rewrite Question: {generated_rewrite_text!r}")
            print('-----------------------')
            print(f"2. Rerank Context: {contents!r}")
            print('-----------------------')
            print(f"3. LLM Prompt: {SYSTEM_PROMPT!r}")
            print('-----------------------')
            print(f"4. Generated text: {generated_model_text!r}")
            print(f"5. Page: {final_answer!r}")
            
        return final_answer.replace('\n', '⁂')