import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
import datetime
import pandas as pd
import boto3
from utils import *
import easyocr
from io import StringIO, BytesIO
from sentence_transformers import SentenceTransformer, util

reader = easyocr.Reader(lang_list=['id']) # easyocr
embedder1 = SentenceTransformer('paraphrase-MiniLM-L3-v2')
embedder2 = SentenceTransformer('sentence-transformers/LaBSE')

st.title('CoLearn Image Search')

uploaded_image = st.file_uploader("Choose an image")
option = st.selectbox('Image Subject', ('Choose', 'Maths', 'Physics', 'Chemistry'))

if (uploaded_image is not None) and (option!='Choose'):
    st.image(uploaded_image, caption='')

    text = extract_text(uploaded_image.getvalue(), reader)
    extracted_full_text = ' '.join(text[1])
    st.write(extracted_full_text)

    query = [extracted_full_text]


    file_name = f'{option.lower()}_corpus_embeddings' + '.pkl'
    destination_file = corpus_prefix + file_name
    print("bucket : " + corpus_bucket + " key : " + destination_file)
    file_corpus_embeddings = BytesIO()
    client.download_fileobj(corpus_bucket, destination_file, file_corpus_embeddings)
    file_corpus_embeddings.seek(0)
    corpus_embeddings = np.load(file_corpus_embeddings)

    file_name = f'text_extracted_{option}.csv'
    corpus_file = corpus_prefix + file_name
    print("bucket : " + corpus_bucket + " key : " + corpus_file)
    obj = client.get_object(Bucket= corpus_bucket, Key= corpus_file) 
    corpus = pd.read_csv(obj['Body'])

    print("corpus shape : " + str(corpus.shape))
    print("corpus embeddings shape : " + str(corpus_embeddings.shape))

    query_embeddings1 = embedder1.encode(query, show_progress_bar=True, convert_to_tensor=False, normalize_embeddings=True)
    query_embeddings2 = embedder2.encode(query, show_progress_bar=True, convert_to_tensor=False, normalize_embeddings=True)
    query_embeddings = np.hstack((query_embeddings1, query_embeddings2))

    print(len(query_embeddings))

    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k = 10)

    query['recommendations'] = hits
    query['recommendation_question_id'] = query['recommendations'].apply(lambda x: [corpus['questionId'].values[i['corpus_id']] for i in x])

    