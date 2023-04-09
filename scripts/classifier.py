from sklearn import tree
import pandas as pd

class Algo:
  """set up and run decision tree algorithm"""
  
  def label_accepted(self, data, hires):
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
  def train(self, X_train, X_test, y_train, y_test):
      print(' ')
      print('Training algorithm... \n')
      classifier = tree.DecisionTreeClassifier(max_depth=5)
      classifier = classifier.fit(X_train, y_train)
      dot_data = tree.export_graphviz(classifier, out_file=None, impurity=False) 
      graphviz.Source(dot_data)
      return [classifier, X_test, y_test, y_train]
  def test(self, classifier, X_test, y_test, y_train):
      print('Testing algorithm... \n')
      y_predict = classifier.predict(X_test)
      print('Algorithm tested. Results obtained. /n')
      return y_predict, X_test, y_test, y_train