# 사용자의 잘문과 유사문서를 검색하기 위한 검색기 코드

from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import numpy as np
from chatbotGUI.models_loader import PREPROCESSED_DOCS, VECTORSTORE, MODEL_RERANKER, TOKENIZER_RERANKER
import torch
import tensorflow.compat.v1 as tf1
import os

# GPU 번호 지정 '0'은 1번 GPU,'1'은 2번 GPU
#nvidia-smi 명령어를 터미널에서 사용하여 사용가능한 GPU의 넘버를 확인할 것
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
config = tf1.ConfigProto()

# GPU 사용량 지정 0.03은 3%를 사용하겠다는 뜻
# 가능한 3% 이하로 사용할 것
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf1.Session(config=config)

def rerank_retriever(query):    
    # BM25 리트리버 설정
    bm25_retriever = BM25Retriever.from_documents(PREPROCESSED_DOCS)
    bm25_retriever.k = 30 
    
    # Chroma 리트리버 설정
    chroma_retriever = VECTORSTORE.as_retriever(search_kwargs={'k':30})

    # Ensemble 리트리버 설정
    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, chroma_retriever],
                        weight={0.7, 0.3})

    docs = ensemble_retriever.invoke(query)
    
    
    pairs = []
    # 문서 포맷팅
    for doc in docs:
        format_docs = (f"Context: {doc.page_content}").replace("\xa0\xa0", " ").replace("\n", "")
        page = doc.metadata.get('page')
        source = doc.metadata.get('source').replace("/home/hjkll25/LM_Project/UH6/", "")
        metadata = f"{source}의 {page} 페이지"
        
        pairs.append([query, f"{metadata}{format_docs}"])
        
# --------------------------------------------------------------------------------------

#   rerank
    def exp_normalize(x):
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()

    MODEL_RERANKER.eval()

    with torch.no_grad():
        # 입력 데이터를 GPU로 이동
        inputs = TOKENIZER_RERANKER(pairs,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt',
                                    max_length=512)

        # 모든 입력 데이터를 GPU로 이동
        inputs = {key: value.to('cuda') for key, value in inputs.items()}

        # 모델 예측 수행
        scores = MODEL_RERANKER(**inputs, return_dict=True).logits.view(-1, ).float()

        # NumPy로 변환하기 전에 CPU로 이동
        scores = exp_normalize(scores.cpu().numpy())

        # 유사도와 문서 정보를 결합
        scored_pairs = list(zip(scores, pairs))

        # 유사도가 0.7 이상인 문서만 필터링
#         filtered_pairs = [(score, doc) for score, doc in scored_pairs if score >= 0.1]
#         sorted_pairs = sorted(filtered_pairs, key=lambda x: x[0], reverse=True)
#         sorted_scores, sorted_docs = zip(*sorted_pairs) if sorted_pairs else ([], [])

        sorted_pairs = sorted(scored_pairs, key=lambda x: x[0], reverse=True)
        top_3_pairs = sorted_pairs[:4]
        sorted_scores, sorted_docs = zip(*top_3_pairs) if top_3_pairs else ([], [])

    return sorted_docs, sorted_scores