import time
import pandas as pd
from scripts import data_partition as tfidf
from scripts import prediction_to_csv as pred2csv
from scripts import text_preprocessing as tp
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Reading the data from the csv files
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Separate them in corpus and labels
corpus_train = tp.stemming_processing(train_df['text'])
labels_train = train_df['label'].to_list()
corpus_test = tp.stemming_processing(test_df['text'])

x_train, y_train, kaggel_list = tfidf.to_TFIDF_bow(corpus_train, labels_train, corpus_test)


#We instanciate the test clasifiers
#clsf = SVC(kernel="rbf")
clsf = MultinomialNB()

# We test the clasifier with 10 fold cross validation
#tfidf.ten_fold_cross_validation_test(corpus_train, labels_train, corpus_test, clsf)

start = time.time()

clsf.fit(x_train, y_train)

stop = time.time()

print("Timpul de antrenare: ", stop-start)
#predict_labels = clsf.predict(kaggel_list)

#pred2csv.write_to_csv(predict_labels)