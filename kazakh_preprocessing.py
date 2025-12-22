import stanza
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

import nltk
nltk.download('punkt_tab')

# The first step in NLP text preprocessing is Tokenization
# The second step is cleaning the raw texts. Removing punctuation marks,
#   removing stopwords(words that we don't need, words without meanings,
#   such as:  “a”, “the”, “is”, “are”, in Kazakh language:
# The third step is word lemmatizing or stemming
# Also, we can extract Part-Of-Speech tagging
# We can use Named Entity Recognition for extraction of the main idea of the text
# Most frequnt nouns
# Most frequent verbs
# Most frequent adjectives

def get_documents_to_preprocess():
    chunk198 = pd.read_csv("chunk198.csv")
    chunk199 = pd.read_csv("chunk199.csv")
    chunk200 = pd.read_csv("chunk200.csv")

    chunk198 = chunk198[chunk198['Language'] == 'Kazakh']
    chunk200 = chunk200[chunk200['Language'] == 'Kazakh']
    dataset = pd.concat([chunk198, chunk199, chunk200])
    documents_to_process = dataset["Text"]
    print("TOTAL docs: ", documents_to_process)

    documents_to_process = list(documents_to_process.iloc[:50])
    return documents_to_process

def get_labeled_data():
    labeled_data = pd.read_csv("sample.csv")
    documents_to_process = list(labeled_data["text"].iloc[:50])

    return documents_to_process

def tokenize_texts(documents):
    tokenized_documents = []
    for text in documents:
        tokenized_documents.append(word_tokenize(text))
    return tokenized_documents

def clean_texts(tokenized_sentences):
    stop_words = stopwords.words('kazakh')
    stop_words.extend(
        ["'", "-", "–", "–", "...", ".", ",", ";", "?", "!", "%", "$", "/", ":", ")",
         "(", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", "-", ".", "/",
         ":", ";", "<", "=", ">", "?", "@", "[", "\"", "]", "^", "_", "`", "{", "|", "}", "~", "”", "“", "—", "»"])

    for i in range(len(tokenized_sentences)):
        for tokenized_sentence in tokenized_sentences:
            for token in tokenized_sentence:
                if token in stop_words:
                    tokenized_sentence.remove(token)
    return tokenized_sentences

def lemmatizing(documents):
    nlp = stanza.Pipeline(lang='kk', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True, download_method=stanza.DownloadMethod.REUSE_RESOURCES)
    lemmatized_documents = []
    doc = nlp(documents)

    for i, sentence in enumerate(doc.sentences):
        lemmas = []
        sentence = sentence.to_dict()
        for token in sentence:
            lemmas.append(token['lemma'])
        lemmatized_documents.append(lemmas)

    return lemmatized_documents


def vectorizing(sentences):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)

    # All words in the vocabulary.
    print("vocabulary", tfidf.get_feature_names_out())
    # IDF value for all words in the vocabulary
    print("IDF for all words in the vocabulary :\n", tfidf.idf_)

    # TFIDF representation for all documents in our corpus
    print('\nTFIDF representation for "{}" is \n{}'
          .format(sentences[0], tfidf_matrix[0].toarray()))
    print('TFIDF representation for "{}" is \n{}'
          .format(sentences[1], tfidf_matrix[1].toarray()))
    print('TFIDF representation for "{}" is \n{}'
          .format(sentences[2], tfidf_matrix[2].toarray()))



if __name__ == '__main__':
    documents = get_documents_to_preprocess()
    tokenized_documents = tokenize_texts(documents)
    cleaned_documents = clean_texts(tokenized_documents)
    print(len(cleaned_documents))

    # [["token", "token", "token", ], "token", "token", "token", ], "token", "token", "token", ], ]
    lemmatized_documents = lemmatizing(cleaned_documents)

    documents_to_vectorize = []
    for document in lemmatized_documents:
        sentence = ""
        for token in document:
            sentence += " " + token
        documents_to_vectorize.append(sentence)

    print("Document: ", documents[20:40])
    print("Tokenized: ", tokenized_documents[20:40])
    print("Cleaned: ", cleaned_documents[20:40])
    print("Lemmatized: ", lemmatized_documents[20:40])
    vectorizing(documents_to_vectorize)




