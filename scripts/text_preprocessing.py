import re
import preprocessor as p
from string import punctuation
from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True) # We declare the tokenizer to strip the sentences of their handels and reduce the length
stemmer = SnowballStemmer("italian") # We initialize the stemmer with words in italian
stop_words = list(stopwords.words('italian'))  # We initialize the stop_word list with most common words in italian that wouldn't change the meaning of the text
stop_words.append(["..", "“", ["’"]]) # To the stop_word list we append a few punctuation symbols that are not in punctuation from string library

# We are tokenizing the strings (using Tweet Tokenizer) then stemming the words and removing the stopwords and punctuation

def recreate_the_sentences(listed_tokens):

    recreated_sentences = []

    for sentence in listed_tokens:
        recreated_sentences.append(' '.join(sentence))

    return recreated_sentences


def stemming_processing(to_be_processed_corpus, recreate = True):

    tokenized_corpus = []

    for sentence in to_be_processed_corpus:
        sentence = p.clean(sentence) # We remove the emoji in the hexadecimal form
        sentence = re.sub(r"http\S+", "", sentence).lower() # We remove the http links from the sentences
        tokenized_sentence = tokenizer.tokenize(sentence) # We tokenize the sentence and remove any tweeter tags
        stemmed_sentence = [] 
        for word in tokenized_sentence:
            if (word not in stop_words) & (word not in punctuation): 
                stemmed_sentence.append(stemmer.stem(word)) # If the word is not a stopword or a punctuation mark we stem it else we remove it
        tokenized_corpus.append(stemmed_sentence) # We add all the stemmed/tokenized sentences to the new corpus
    
    if recreate == True:
        return recreate_the_sentences(tokenized_corpus)
    
    return tokenized_corpus




