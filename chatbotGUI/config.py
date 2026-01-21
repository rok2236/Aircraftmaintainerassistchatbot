from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv, find_dotenv
from langchain_teddynote import logging

def start_langsmith():
    load_dotenv(find_dotenv())
    logging.langsmith("chatbot_test")