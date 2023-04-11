from sklearn import tree
import pandas as pd
import streamlit as st

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
        with st.spinner('Training classifier...'):
            classifier = tree.DecisionTreeClassifier(max_depth=5)
            classifier = classifier.fit(X_train, y_train)
        return (classifier, X_test, y_test, y_train)

    def test(classifier, X_test, y_test, y_train):
        with st.spinner('Testing classifier...'):
            y_predict = classifier.predict(X_test)
        return y_predict, X_test, y_test, y_train