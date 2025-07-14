import nltk
import bs4 as bs 
import re 
import requests
import urllib.request
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import heapq #for selecting the top ranked sentences

article_url = urllib.request.urlopen('https://en.wikipedia.org/wiki/Winter_Is_Coming') #opening the article page

article = article_url.read() # reading the article

parsed_article = bs.BeautifulSoup(article, 'lxml')
paragraphs = parsed_article.find_all('p')

article_text = ' ' # empty string to store text 
for p in paragraphs:
    article_text += p.text
# stores only the text without the p tag

# function that will clean the text (removes punctuation)
def clean_text(article_text):
    formatted_article = re.sub(r'\s+', ' ', article_text) #removes extra spaces
    formatted_article =  re.sub(r'[\'\"“”‘’]', '', formatted_article) #removes quotation marks
    formatted_article = re.sub(r'[^a-zA-Z0-9.,;?!\s]', '', formatted_article) # removes special characters except basic punctuation
    return formatted_article.strip()

cleaned_text = clean_text(article_text)

sentences = nltk.sent_tokenize(cleaned_text)
words = nltk.word_tokenize(cleaned_text)
stop_words = nltk.corpus.stopwords.words('english') 
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
# make functions
word_frequencies = {}
for word in words:
    word = word.lower() # lowercase before removing stopwords 
    if word not in stop_words and word.isalpha(): # alphabetic characters only and removing stopwords 
        stemmed_word = lemmatizer.lemmatize(word) # to avoid repeated frequencies 
        if stemmed_word not in word_frequencies.keys(): # keeps track of how many times each word appears in the text.
            word_frequencies[stemmed_word] = 1
        else:
            word_frequencies[stemmed_word] += 1  
    
sentence_score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
         if word not in stop_words and word.isalpha():
            stemmed_word = lemmatizer.lemmatize(word)
            if stemmed_word in word_frequencies.keys(): #checks if the word is in impt words dictionary
                if len(sentence.split(' ')) < 30: #avoids the overly long and complex sentences 
                    if sentence not in sentence_score.keys():
                        sentence_score[sentence] = word_frequencies[stemmed_word]
                    else:
                        sentence_score[sentence] += word_frequencies[stemmed_word]
                    

# summarising 
summary_sentences = heapq.nlargest(10, sentence_score, key=sentence_score.get)
summary_sentences.sort(key=lambda s: sentences.index(s))  # maintain original order
summary = ' '.join(summary_sentences)
print(summary)