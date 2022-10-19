import os
import re
import numpy as np
import cv2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.cluster.util import cosine_distance
from textblob import TextBlob
import heapq
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import PyPDF2
import pyttsx3 as pyttsx3
import fitz
from PIL import Image
from gensim.parsing import remove_stopwords
import warnings
import pandas as pd
import streamlit as st
def img_to_pdf(doc_file):
    """
        Convert image file to .pdf file if the input file is docx file
        :parameter doc_file: URL of the image file
        :return pdf_loc: URL of the pdf file
    """
    pdf_loc = doc_file.split(".")[0] + ".pdf"
    if doc_file.split(".")[1] == 'png' or doc_file.split(".")[1] == 'jpg' or doc_file.split(".")[1] == 'svg':
        pdf_loc = doc_file.split(".")[0]+".pdf"
        image_1 = Image.open(doc_file)
        im_1 = image_1.convert('RGB')
        im_1.save(pdf_loc)
    return pdf_loc
def get_metadata(doc_file):
    """
        Extract metadata of the document
        :parameter
        doc_file: URL of the docx file
        :return
        metas: list of all the metadata of the document
    """
    open_doc = open(doc_file, 'rb')
    hand_book = PyPDF2.PdfFileReader(open_doc)
    data = hand_book.getDocumentInfo()
    metas = {}
    for metadata in data:
        metas[metadata[1:]] = data[metadata]
    return metas
def get_images(doc_file):
    """
        Extract images from the document and save it in the local system
        :parameter
        doc_file: URL of the docx file
        :return
        img_lst: list of URL of the saved images
    """
    input1 = PyPDF2.PdfFileReader(open(doc_file, "rb"))
    img_lst = []
    pdfpages = input1.getNumPages()
    for i in range(pdfpages):
        page = input1.getPage(i)
        try:
            xObject = page['/Resources']['/XObject'].getObject()

            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                    data = xObject[obj].getData()
                    if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                        mode = "RGB"
                    else:
                        mode = "P"
                    if xObject[obj]['/Filter'] == '/FlateDecode':
                        img = Image.frombytes(mode, size, data)
                        img.save(obj[1:] + ".png")
                        img_lst.append(obj[1:] + ".png")
                        print("Image found in page", i+1, "Saving ...")
                    elif xObject[obj]['/Filter'] == '/DCTDecode':
                        img = open(obj[1:] + ".jpg", "wb")
                        img.write(data)
                        img.close()
                        img_lst.append(obj[1:] + ".jpg")
                    elif xObject[obj]['/Filter'] == '/JPXDecode':
                        img = open(obj[1:] + ".jp2", "wb")
                        img_lst.append(obj[1:] + ".jp2")
                        img.write(data)
                        img.close()
        except:
            print("No image in page", i+1)
    return img_lst
def show_images(img_lst):
    for x in img_lst:
        img = cv2.imread(x, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def extract_text(doc_file):
    """
    Extract text from the document, process it and return list of sentences in the document
    :parameter
    doc_file: URL of the document
    :return
    all_text: List of all the sentences in the document
    """
    doc = fitz.open(doc_file)
    all_text = chr(12).join([page.get_text() for page in doc])
    all_text = all_text.split(".")
    l = len(all_text)
    x = 0
    while x < l:
        all_text[x] = (all_text[x].replace('\n', ''))+'.'
        if all_text[x].strip(' ') == ".":
            all_text.remove(all_text[x])
            l -= 1
        else:
            x += 1
    return all_text

def number(doc_file):
    open_doc = open(doc_file, 'rb')
    hand_book = PyPDF2.PdfFileReader(open_doc)
    return hand_book.numPages

def get_author(doc_file):
    open_doc = open(doc_file, 'rb')
    hand_book = PyPDF2.PdfFileReader(open_doc)
    try:
        return hand_book.getDocumentInfo()["/Author"]
    except:
        return None

def get_creation_date(doc_file):
    open_doc = open(doc_file, 'rb')
    hand_book = PyPDF2.PdfFileReader(open_doc)
    cdate = (hand_book.getDocumentInfo()['/CreationDate'])
    cdate = cdate[2:6] + "-" + cdate[6:8] + "-" + cdate[8:10]
    return cdate
def hear(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return
def keyword_extraction(text):
    """
    Extract only necessary words from the document
    :param text: List of sentences in document
    :return keywords: List of cleaned words
    """
    text = ' '.join([w.lower() for w in text])
    stop_words = stopwords.words('english')
    re.sub("\s+", " ", text)
    re.sub(r'[^\w ]+', "", text)
    words = nltk.word_tokenize(text)
    punctuations = re.sub(r'\W', ' ', str(text))
    num_less = re.sub('\w*\d\w*', '', text).strip()
    keywords = [word for word in words if not word in stop_words and word in punctuations and word in num_less]
    return keywords
def make_df(doc_text):
    keywords = keyword_extraction(doc_text)
    data = ' '.join(keywords)
    df = pd.DataFrame([data])
    df.columns = ['Script']
    df.index = ['Line']
    return df
def bow(doc_text):
    """
    Generate Bag of Words from the dataframe
    :param doc_text: List of sentences of the document
    :return data: bow dataframe
    """
    df = make_df(doc_text)
    corpus = df.Script
    cv = CountVectorizer(stop_words='english')
    data = cv.fit_transform(corpus)
    feature_names = cv.get_feature_names()
    data = pd.DataFrame(data.toarray(), columns=feature_names)
    data.index = df.index
    return data
def most_frequent(doc_text):
    """
    Gives most frequent words in the document
    :param doc_text: List of sentences of the document
    :return tops: frequent words
    """
    data = bow(doc_text)
    data = data.transpose()
    tops = []
    for c in data.columns:
        top = data[c].sort_values(ascending=False)
        tops = list(zip(top.index, top.values))
    return tops
def summarize(doc_text):
    """
    Provides Summary of the document
    :param doc_text: List of sentences of the document
    :param n: Number of lines
    :return summary: Document summary
    """
    x = len(doc_text)
    n = 1
    if x >= 100:
        n = 10
    elif x >= 50:
        n = 7
    elif x >= 5:
        n = 4
    else:
        n = x
    lower_text = ' '.join([w.lower() for w in doc_text])
    formatted = re.sub('[^a-zA-Z]', ' ', lower_text)
    formatted = re.sub(r'\s+', ' ', formatted)
    l = [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(formatted)]
    l = " ".join(l)
    sentence_list = doc_text
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(l):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    summary_sentences = heapq.nlargest(n, sentence_scores, key=sentence_scores.get)
    summary = '\n\n'.join(summary_sentences)
    return summary
def lemmatization(doc_text):
    txt = " ".join(doc_text)
    return [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(txt)]
def topic_modelling(doc_text):
    text = ' '.join([w.lower() for w in doc_text])
    stop_words = stopwords.words('english')
    words = nltk.word_tokenize(text)
    punctuations = re.sub(r'[^\w ]+', "", text)
    num_less = re.sub('\w*\d\w*', '', text).strip()
    keywords = [word for word in words if not word in stop_words and word in punctuations and word in num_less]
    txt = " ".join(keywords)
    l = [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(txt)]
    lemms = []
    lemms.append(l)
    id2word = corpora.Dictionary(lemms)
    texts = lemms
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, alpha='auto', num_topics=20,
                                                random_state=100,
                                                update_every=1, passes=20, per_word_topics=True)
    topics = lda_model.print_topics()
    topic = []
    s = ""
    for i in topics:
        temp = []
        for x in i:
            s = x
        lst = s.split("+")
        for j in lst[0:-2]:
            temp.append(str(j).split("*")[1][1:-2])
        temp.append(str(lst[-1]).split("*")[1][1:-1])
        topic.append(temp)
    return topic
def sentence_similarity(sent1, sent2):
    """
    Identifies similarity between two sentences
    """
    stop_words = stopwords.words('english')
    sent1 = sent1.split()
    sent2 = sent2.split()
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    every = list(set(sent1 + sent2))
    vec1 = [0] * len(every)
    vec2 = [0] * len(every)
    for w in sent1:
        if w in stop_words:
            continue
        vec1[every.index(w)] += 1
    for w in sent2:
        if w in stop_words:
            continue
        vec2[every.index(w)] += 1
    return 1 - cosine_distance(vec1, vec2)
def polarity(words):

    text = ' '.join(words)
    val = TextBlob(text).sentiment[0]
    polar_val = 'Positive'
    if val >= -0.5 and val <= 0.5 :
        polar_val = 'Neutral'
    elif val < -0.5:
        polar_val = 'Negative'
    return [val, polar_val]
def subjectivity(words):
    text = ' '.join(words)
    val = TextBlob(text).sentiment[1]
    sub_val = 'Subjective'
    if val < 0.5:
        sub_val = 'Objective'
    return [val, sub_val]
def pos_tag(doc_text):
    """
    Performs Parts Of Speech tagging
    :param doc_text: List of sentences in the document
    :return: List with POS tagging of tokenized words
    """
    text = ' '.join([w.lower() for w in doc_text])
    pos = []
    tokens = word_tokenize(text)
    for word in tokens:
        pos.append(nltk.pos_tag([word]))
    return pos
def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence
def get_cleaned_sentences(tokens, stopwords=False):
    cleaned_sentences = []
    for line in tokens:
        cleaned = clean_sentence(line, stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences
def TFIDF_Q(question_to_be_cleaned, cleaned_sentences):
    tfidfvectoriser = TfidfVectorizer()
    tfidfvectoriser.fit(cleaned_sentences)
    tfidf_vectors= tfidfvectoriser.transform([question_to_be_cleaned])
    return tfidf_vectors
def preprocessing(txt):
    nltk.download('punkt')
    tokens = nltk.sent_tokenize(txt)
    tfidfvectoriser=TfidfVectorizer()
    cleaned_sentences = get_cleaned_sentences(tokens, stopwords=True)
    cleaned_sentences_with_stopwords =get_cleaned_sentences(tokens, stopwords=False)
    tfidfvectoriser.fit(cleaned_sentences)
    tfidf_vectors = tfidfvectoriser.transform(cleaned_sentences)
    return [cleaned_sentences, cleaned_sentences_with_stopwords, tfidf_vectors]
def Euclidean(question_vector, sentence_vector):
    vec1 = question_vector.copy()
    vec2 = sentence_vector.copy()
    if len(vec1) < len(vec2): vec1, vec2 = vec2, vec1
    vec2 = np.resize(vec2, (vec1.shape[0], vec1.shape[1]))
    return np.linalg.norm(vec1 - vec2)
def answer(question_vector, sentence_vector):
        return Euclidean(question_vector, sentence_vector)
def retrieve_answer(question_embedding, tfidf_vectors, method=1):
    similarity_heap = []
    for index, embedding in enumerate(tfidf_vectors):
        similarity =answer((question_embedding).toarray(), (embedding).toarray()).mean()
        heapq.heappush(similarity_heap, (similarity, index))
    return similarity_heap
def search_ans(doc_text, question):
    """
    Search answer for the query in the document
    :param doc_text: List of sentences of the document
    :param question: Question
    :return: Suitable answer for the question
    """
    txt = ' '.join([w.lower() for w in doc_text])
    preprocess = preprocessing(txt)
    cleaned_sentences, cleaned_sentences_with_stopwords, tfidf_vectors = preprocess
    question = clean_sentence(question, stopwords=True)
    question_embedding = TFIDF_Q(question, cleaned_sentences)
    similarity_heap = retrieve_answer(question_embedding, tfidf_vectors)
    number_of_sentences_to_print = 2
    answers = []
    while number_of_sentences_to_print > 0 and len(similarity_heap) > 0:
        x = similarity_heap.pop(0)
        answers.append(cleaned_sentences_with_stopwords[x[1]])
        number_of_sentences_to_print -= 1
    return answers
def save_uploadedfile(uploadedfile):
    with open(os.path.join(os.getcwd() , uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return uploadedfile.name
def st_ui():
    st.title("Document digitization")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    datafile = st.file_uploader(label='Your document will be processed', type=['png', 'jpg', 'pdf'],
                                accept_multiple_files=False, label_visibility="visible")
    if datafile is not None:
        file_details = {"FileName": datafile.name, "FileType": datafile.type}
        datafile = save_uploadedfile(datafile)
    else:
        datafile = "demo.pdf"
    doc_file = img_to_pdf(datafile)
    author = str(get_author(doc_file))
    st.text("Author of the Document : " + author)
    creationDate = get_creation_date(doc_file)
    st.text("Date of Creation " + creationDate)
    total_pages = number(doc_file)
    st.text("Document metadata :")
    meta_data = get_metadata(doc_file)
    st.text(meta_data)
    st.text("Extracting images in the document ...")
    img_lst = get_images(doc_file)
    # show_images(img_lst)
    for x in img_lst:
        img = cv2.imread(x, cv2.IMREAD_ANYCOLOR)
        st.image(img, width=200)
    st.text("Extracting text in the document ...")
    doc_text = extract_text(doc_file)
    st.text(doc_text)
    keywords = keyword_extraction(doc_text)
    df = make_df(doc_text)
    bag_of_words = bow(doc_text)
    t = bag_of_words.transpose()
    st.text("Bag of Words :")
    st.text(t)
    top100 = most_frequent(doc_text)
    st.text("Most frequent words in document :")
    st.text(top100)
    st.text("Polarity :")
    st.text(polarity(keywords))
    st.text("Subjectivity :")
    st.text(subjectivity(keywords))
    st.text(sentence_similarity("Hello I am a document reader", "Welcome to document reader show"))
    st.text("Summary of the document :")
    summary = summarize(doc_text)
    st.text(summary)
    topic = topic_modelling(doc_text)
    st.text("Topic modelling :")
    st.text(topic)
    st.text("Lemmas of the document :")
    st.text(lemmatization(doc_text))
    st.text("POS tagging :")
    st.text(pos_tag(doc_text))
    st.text("Question answer finding :")
    question = st.text_input("Enter your query to find an answer from the document")
    st.text(search_ans(doc_text, question))
if __name__ == "__main__":
    st_ui()
