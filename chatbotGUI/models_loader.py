#  필요 모델을 모두 불러오는 코드
import pickle
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from vllm import LLM, SamplingParams
import torch
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


# 다른 곳에서 모델을 사용하기 위한 전역 변수
PREPROCESSED_DOCS = None
VECTORSTORE = None
LLM_MAIN = None
SAMPLING_PARAMS_MAIN = None
LLM_REWRITE = None
SAMPLING_PARAMS_REWRITE = None
TOKENIZER_REWRITE = None
MODEL_RERANKER = None
TOKENIZER_RERANKER = None
EMBEDDINGS_MODEL = None

CACHE_DIR = "/home/nas/data/hjkll25/models"

def initialize_models():
    global LLM_MAIN, SAMPLING_PARAMS_MAIN
    global VECTORSTORE, LLM_REWRITE, SAMPLING_PARAMS_REWRITE
    global TOKENIZER_REWRITE, MODEL_RERANKER, TOKENIZER_RERANKER, EMBEDDINGS_MODEL, PREPROCESSED_DOCS

    # Embeddings 모델 초기화
    EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-base',
        cache_folder=CACHE_DIR,
        encode_kwargs={'normalize_embeddings': True}
    )

    # FAISS 벡터스토어 로드
    VECTORSTORE_PATH = '/home/nas/data/hjkll25/db/percentile'
    VECTORSTORE = FAISS.load_local(
        VECTORSTORE_PATH,
        EMBEDDINGS_MODEL,
        allow_dangerous_deserialization=True
    )

    # EuroLLM 모델
    MODELTORUN_REWRITE = "/home/nas/data/hjkll25/models/models--utter-project--EuroLLM-9B-Instruct/snapshots/5993d62643602953b89c9fed93970a37aedc3ac7"
    LLM_REWRITE = LLM(model=MODELTORUN_REWRITE,
         enforce_eager=True,
         dtype=torch.bfloat16,
         max_model_len=3088,
         gpu_memory_utilization=0.5,
         speculative_max_model_len = 30624,
         max_num_batched_tokens=4000,
         max_seq_len_to_capture=4000,
         trust_remote_code=True,
         device="cuda")
    SAMPLING_PARAMS_REWRITE = SamplingParams(temperature=0.3, top_p=0.85, max_tokens=5000)
    TOKENIZER_REWRITE = AutoTokenizer.from_pretrained("utter-project/EuroLLM-9B-Instruct", cache_dir=CACHE_DIR, trust_remote_code=True)

    # Orion 모델
    MODELTORUN = "/home/nas/data/hjkll25/models/models--OrionStarAI--Orion-14B-Chat-Int4/snapshots/c6145085292d31ccab733c82a7c10bad43addade"
    LLM_MAIN = LLM(model=MODELTORUN,
         enforce_eager=True,
         dtype=torch.bfloat16,
        #  gpu_memory_utilization=0.5,
         max_model_len=4096,
        #  seed=415,
         speculative_max_model_len = 10000,
         max_num_batched_tokens=5000,
         max_seq_len_to_capture=1000,
         trust_remote_code=True,
         device="cuda")
    SAMPLING_PARAMS_MAIN = SamplingParams(temperature=0.8, top_p=0.85, max_tokens=5000)

    # ReRank 모델
    MODEL_RERANKER_DIR = "Dongjin-kr/ko-reranker"
    MODEL_RERANKER = AutoModelForSequenceClassification.from_pretrained(
        MODEL_RERANKER_DIR,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    TOKENIZER_RERANKER = AutoTokenizer.from_pretrained(MODEL_RERANKER_DIR, cache_dir=CACHE_DIR, trust_remote_code=True)

    # 파일에서 전처리된 문서 불러오기
    with open('/home/hjkll25/LM_Project/vllm/preprocessed_docs.pkl', 'rb') as file:
        PREPROCESSED_DOCS = pickle.load(file)

    print("모델과 데이터 초기화 완료!")
