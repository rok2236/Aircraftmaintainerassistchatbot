# 검색기, 생성기 기능 하나의 함수로 만듬
# views.py에서 사용자의 질문을 받아 입력으로 넣어줌
def gen_answer(q):
    from chatbotGUI.retriever import load_vdb, retriever, load_embedding_models
    from chatbotGUI.generater import generate_answer

    embeddings_models = load_embedding_models()
    vdb = load_vdb(embeddings=embeddings_models,
            vec_dir="/home/nas/data/hjkll25/test_db")


    query = q
    retriever_docs = retriever(vectorstore=vdb, query=query)


    gen_ans = generate_answer(query=query, docs=retriever_docs)
    
    
    return gen_ans

