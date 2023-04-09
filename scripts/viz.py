import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz
import streamlit

class UI:
  """showcase results of program through graphs"""
  
  def bar_graph(self, accepted_rejected_data, df, graph_type):
    df['Accepted_Rejected'] = accepted_rejected_data
    accepted_men, rejected_men, accepted_women, rejected_women = (0,0,0,0)
    for row in df.itertuples():
      if row.Gender == 0 and row.Accepted_Rejected == 'Accepted':
        accepted_men += 1
      elif row.Gender == 0 and row.Accepted_Rejected == 'Rejected':
        rejected_men += 1
      elif row.Gender == 1 and row.Accepted_Rejected == 'Accepted':
        accepted_women += 1
      elif row.Gender == 1 and row.Accepted_Rejected == 'Rejected':
        rejected_women += 1
      else:
        raise TypeError
    N = 2
    men_data = [accepted_men, rejected_men]
    women_data = [accepted_women, rejected_women]
    width = .25
    ind = np.arange(N)       
    plt.bar(ind, men_data, width, label='Men')
    plt.bar(ind + width, women_data, width,
        label='Women')
    plt.ylabel('Number of People')
    plt.title('Accepted/Rejected Applicants by Gender \n' + graph_type)
    plt.xticks(ind + width / 2, ('Accepted', 'Rejected'))
    plt.legend(loc='best')
    plt.show()

    def run(self):
        """engage the multiple parts of the program"""
        
        template = create.data
        biased_dataframe, bias_data = create.biased_dataset(template)
        unbiased_dataframe, no_bias_data = create.unbiased_dataset(template)
        choice_print = input('Print generated datasets? (Y/N): ').upper()
        if choice_print == 'Y':
        print('Biased data: \n \n', biased_dataframe, '\n')
        print('Unbiased data: \n \n', unbiased_dataframe, '\n')
        prep = True
        test_vars = []
        while prep:
        training_data = input('Should a biased dataset be used to train the algorithm? (Y/N): ').upper()
        if training_data == 'Y':
            prep = False
            test_vars=functions.train(bias_data[0], no_bias_data[0], bias_data[1], no_bias_data[1])
        elif training_data == 'N':
            prep = False
            test_vars=functions.train(no_bias_data[0], bias_data[0], no_bias_data[1], bias_data[1])
        else:
            print('Error')
        y_predict, X_test, y_test, y_train = functions.test(test_vars[0], test_vars[1], test_vars[2], test_vars[3])
        print('')
        print('Analyzing results... \n')
        print("The first graph shows the ideal, unbiased hiring decisions. The second shows the algorithm's decisions. \n")
        graph.bar_graph(y_train, biased_dataframe, "Algorithm's Training Data (Biased)")
        graph.bar_graph(y_test, unbiased_dataframe, 'Pre-Generated Unbiased Hiring')
        graph.bar_graph(y_predict, unbiased_dataframe, 'Algorithm Generated Hiring')