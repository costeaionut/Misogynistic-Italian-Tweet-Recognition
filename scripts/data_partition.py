import numpy as np
from statistics import mean 
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

mean_accuracy = []
matrix_to_confuse = []

# Definition to transform from sentences to bag of words using TF/IDF

def to_TFIDF_bow(corpus_train, train_labels, corpus_kaggel):

    corpus_train, train_labels = shuffle(corpus_train, train_labels) # We shuffle the train data for a better learning

    vectorizer = TfidfVectorizer(max_features = 1500) # We vectorize the data using the TF/IDF
    
    train_data_list = vectorizer.fit_transform(corpus_train).todense().tolist() # Transform the data to list. We use fit transform for train to insert all the data
    kaggel_test_list = vectorizer.transform(corpus_kaggel).todense().tolist() # Transform the data to a list. We use transform for test to append all the new data

    return train_data_list, train_labels, kaggel_test_list


# Definition of the ten fold cross validation

def ten_fold_cross_validation_test(corpus_train, labels_train, corpus_test, clasifier):
    
    X, y, kaggel_list = to_TFIDF_bow(corpus_train, labels_train, corpus_test) # We process the data so it can be tested

    kf = KFold(n_splits = 10, shuffle = True) #Declare the ten fold cross validation

    idx = 0

    for train_index, test_index in kf.split(X):

        idx = idx + 1
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        cls = clasifier
        
        for train_idx, test_idx in zip(train_index, test_index):
            x_train.append(X[train_idx])
            x_test.append(X[test_idx])

            y_train.append(y[train_idx])
            y_test.append(y[test_idx])

        cls.fit(x_train, y_train)
        predict_labels = cls.predict(x_test)

        mean_accuracy.append(accuracy_score(predict_labels, y_test))
        matrix_to_confuse.append(confusion_matrix(predict_labels, y_test))

        print("\nIteratia numarul ", idx)
        print("Acuratetea modelului este: ", accuracy_score(predict_labels, y_test))
    
    print("\nMedia acuratetii pe model: ", mean(mean_accuracy))
    print("Matricea de confuzie este: \n", sum(matrix_to_confuse))

