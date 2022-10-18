# Document Processing AI
Document Processing AI can extract metadata, text and image from the document in .pdf .jpg .png .svg format
It can be used to perform NLP tasks: 
Text cleaning, Lemmatization, POS tagging , Bag of Words, Most frequent words, Answer retrieval , Text summarization, Topic modelling, Sentence Similarity
# How to call it in your Python code
```
import pydaisi as pyd
document_processing_ai = pyd.Daisi("goldenqueen/Document Processing AI")
```
To convert Image into PDF file
```
doc_file = document_processing_ai.img_to_pdf(<URL of your document>).value
```
To extract Metadata of the document pass the URL of the document to the following function
```
meta_data = document_processing_ai.get_metadata(doc_file).value
```
To extract text from the document
```
doc_text = document_processing_ai.extract_text(doc_file).value
```
To extract Images present in the document
```
img_list = document_processing_ai.get_images(doc_file).value
```
To extract meaningful words from the document
```
keywords = document_processing_ai.keyword_extraction(doc_text).value
```
To clean the document
```
cleaned_sentences_list = document_processing_ai.get_cleaned_sentences(tokens, stopwords=None).value
```
To generate Bag of Words for the document
```
bag_of_words_list = document_processing_ai.bow(doc_text).value
```
To get Summary of the document you can pass doc_text and number of lines of summary needed
```
summary = document_processing_ai.summarize(doc_text, n=4).value
```
For topic modeling
```
topics = document_processing_ai.topic_modelling(doc_text).value
```
To search for an answer in the document
```
answer = document_processing_ai.search_ans(doc_text, question).value
```
To get most frequent words in the document
``` 
most_frequent = document_processing_ai.most_frequent(doc_text).value
```
To find Euclidean distance between TFIDF vector of two sentences
``` 
dist = document_processing_ai.Euclidean(question_vector, sentence_vector).value
```
