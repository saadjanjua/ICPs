from bs4 import BeautifulSoup
import requests
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from collections import Counter
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
url = "https://en.wikipedia.org/wiki/Google"
html = requests.get(url)
web_content = html.content
soup = BeautifulSoup(web_content, "html.parser")
text_content = soup.text

# remove linefeeds and punctuations
# print(lem_words)
lines = text_content.split("\n")
words = ' '.join([word for word in lines if word != ""])
wtokens_p = nltk.word_tokenize(words)
#remove punctuation

#clean_text = ' '.join([word for word in wtokens_p if word.isalpha()])

with open('nlp_output.txt', 'a+', encoding='utf-8') as file:
    file.write(words)

stokens = nltk.sent_tokenize(words)
wtokens = nltk.word_tokenize(words)

w_punctokens = wordpunct_tokenize(words)

clean_wtokens = [word for word in wtokens if word.isalpha()]

for s in stokens:
    print(s)

for w in wtokens:
    print(w)

# print pos_tag
print(nltk.pos_tag(wtokens))

pStemmer = PorterStemmer()
lStemmer = LancasterStemmer()
sStemmer = SnowballStemmer('english')
for x in clean_wtokens:
    print(pStemmer.stem(x))
    print(lStemmer.stem(x))
    print(sStemmer.stem(x))

lemmatizer = nltk.stem.WordNetLemmatizer()
print("\n Lemmatization results")
for x in clean_wtokens:
    print(lemmatizer.lemmatize(x))
#find the trigrams
trigrams = ngrams(clean_wtokens, 3)

# find frequency of trigrams
trigram_freq = Counter(trigrams)
print("\n")
# print(trigram_freq)
# Print top 10 frequent trigrams
top_10 = trigram_freq.most_common(10)

print(top_10)

#NER
print("\n NER")
print(ne_chunk(pos_tag(w_punctokens)))

print(top_10)