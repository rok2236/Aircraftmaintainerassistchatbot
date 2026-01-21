# Aircraft maintainer assist chatbot
A chatbot to assist aircraft maintenance technicians.

# Dataset

I used the UH 60 maintenance document. Please download and use the document from the Google Drive indicated when creating the Vector DB and pkl.
Due to the large size of the PDF file used, I'm uploading it via a Google Drive link. The PDF is a Korean translation of the UH-60 helicopter maintenance manual. The translation was done using a PDF translator.

https://drive.google.com/file/d/13c9v9xVYOyw6GkE04f1yFWIsDRgC_39W/view?usp=sharing


# How to use

1. Create pkl files for VectorDB and BM25 searcher for similarity search using the 00_VectorDB_pkl_maker.ipynb code in the DBmaker folder.
  You must specify the location of the model, the DB, and the path to store the pkl file. You can set the chunk size by modifying the min_chunk_size value in the text_splitter of create_vector_store (default: 150).
  Please specify a folder containing PDFs, not a PDF file, as PDF Loader reads all PDFs in the specified folder.

3. Specify the paths to the models, database, and pkl required for the models loader in the chatbotGUI. If the model isn't in the path, it will be downloaded automatically.

4. You can adjust the weight ratios of the VectorDB and BM25 searchers in retriever.py inside the chatbotGUI. You can specify the number of reranked documents to be included in the model that generates answers from the two searchers by replacing n with a number in the line top_3_pairs = sorted_pairs[:n] at the bottom (it's 4 in the code). Alternatively, you can filter the results by similarity score by replacing it with the commented code above. If filtering by score, comment out the part that filters by the top count.

5. You can set prompts and model parameters in generater.py inside chatbotGUI.

6. In views.py inside chatbotGUI, we pass the question entered from the django web page to the model.

7. To run it, navigate to the location of manage.py in the terminal and run "python manage.py runserver --noreload" . After running it, go to the activated web and send a chat message to confirm the operation. --noreload prevents the model from being loaded onto the GPU twice.
