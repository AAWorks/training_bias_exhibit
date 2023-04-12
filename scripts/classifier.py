from sklearn import tree
import pandas as pd
import streamlit as st
import time

class AlgoUtils:
    """set up and run decision tree algorithm"""

    def label_accepted(data, hires):
        score = []
        accepted = []
        for row in data.itertuples():
            avg = sum([row.Skill, row.Academics, row.Experience, row.Ambition]) / 4
            score.append(avg)
        sorted_score = score
        sorted_score.sort()
        accepted_scores = sorted_score[(hires * -1):]
        for row in data.itertuples():
            avg = sum([row.Skill, row.Academics, row.Experience, row.Ambition]) / 4
            if accepted.count('Accepted') <= hires and avg in accepted_scores:
                accepted.append('Accepted')
            else:
                accepted.append('Rejected')
        return accepted

    def train(X_train, X_test, y_train, y_test):
        clf = tree.DecisionTreeClassifier(max_depth=5)
        classifier = clf.fit(X_train, y_train)
        my_bar = st.progress(0, text="Training Classifier...")
        for percent_complete in range(100):
            time.sleep(0.03)
            my_bar.progress(percent_complete / 100, text="Training Classifier...")
        my_bar.progress(1.0, text="Classifier Trained :white_check_mark:")
        return (classifier, X_test, y_test, y_train), clf

    def test(classifier, X_test, y_test, y_train):
        with st.spinner('Testing classifier...'):
            y_predict = classifier.predict(X_test)
        return y_predict, X_test, y_test, y_train