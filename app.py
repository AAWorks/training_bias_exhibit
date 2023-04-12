import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz

from scripts.generator import TestSet
from scripts.classifier import AlgoUtils

class UI:
    bias: str
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Automated Hiring Simulator", page_icon=":man_in_tuxedo:")
        st.title("Exhibiting Training Bias in Simple Automated Hiring")
        self.bias = "Male"
    
    def __generator_inputs(self):
        with st.form(key='generator'):
            st.write("Dataset Generator Inputs")
            col0, col1, col2 = st.columns(3)
            applicants = col0.number_input("Number of Applicants", min_value = 100, value=10000, step=100)
            hires = col1.number_input("Number of Hires", max_value=applicants, min_value=50, value=500, step=100)
            self.bias = col2.selectbox(
                'Data Skew',
                ("Male", "Female"))
            submit = st.form_submit_button(label='Generate', use_container_width=True)
        if submit:
            return (int(applicants), int(hires), self.bias)

    def __display_datasets(self, df1, df2, data1, data2):
        df1['Status'] = data1[1]
        df2['Status'] = data2[1]
        with st.expander("Generated Biased Data"):
            st.dataframe(df1, use_container_width = True)
        with st.expander("Generated Unbiased Data"):
            st.dataframe(df2, use_container_width = True)

    def run_generator(self):
        inputs = self.__generator_inputs()
        if inputs:
            dataset = TestSet(*inputs)
            biased_dataframe, bias_data = dataset.biased_dataset()
            unbiased_dataframe, no_bias_data = dataset.unbiased_dataset()
            self.__display_datasets(biased_dataframe, unbiased_dataframe, bias_data, no_bias_data)
            return (biased_dataframe, bias_data, unbiased_dataframe, no_bias_data)
        return tuple()
    
    def run_classifier(self, data: tuple):
        _, bias_data, _, no_bias_data = data
        test_vars, clf = AlgoUtils.train(bias_data[0], no_bias_data[0], bias_data[1], no_bias_data[1])
        return AlgoUtils.test(*test_vars), clf
    
    def __show_classifier(self, clf):
        dot_data = tree.export_graphviz(clf, out_file=None, 
                                filled=True)

        graph = graphviz.Source(dot_data, format="png") #work in progress
        st.image(graph, caption="Decision Tree Classifier")
    
    def __bar_graph(self, accepted_rejected_data, df, graph_type):
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
        N = 2
        men_data = [accepted_men, rejected_men]
        women_data = [accepted_women, rejected_women]
        width = .25
        ind = np.arange(N)
        fig, ax = plt.subplots()    
        ax.bar(ind, men_data, width, label='Men')
        ax.bar(ind + width, women_data, width,
                label='Women')
        plt.ylabel('Number of People')
        plt.title('Accepted/Rejected Applicants by Gender \n' + graph_type)
        plt.xticks(ind + width / 2, ('Accepted', 'Rejected'))
        plt.legend(loc='best')
        _, col, _ = st.columns([1,3,1])
        col.pyplot(fig, use_container_width = False)
    
    def display_analytics(self, y_train, y_test, y_predict, data, clf):
        biased_dataframe, unbiased_dataframe = data[0], data[2]
        train, pregen, algogen, algo = st.tabs(["Generated Biased Hiring Data",
                                          "Generated Unbiased Hiring Data",
                                          "Algorithm's Hiring Decisions Given The Unbiased Data",
                                          "Decision Tree Graphic"])
        with train:
            self.__bar_graph(y_train, biased_dataframe, "Generated Biased Hiring Data")
            st.info(f'The generated biased dataset is plotted above. Acceptances and rejections are determined \
                       based off the average of the four major skill categories in the dataset. As can be seen, \
                       the dataset has a skew towards the {self.bias} gender as {self.bias.lower()}s are given \
                       a slightly higher average point for each major stat.')
        with pregen:
            self.__bar_graph(y_test, unbiased_dataframe, 'Generated Unbiased Hiring Data')
            st.info(f'The generated unbiased dataset is plotted above. Acceptances and rejections are determined \
                       based off the average of the four major skill categories in the dataset. As can be seen, \
                       the dataset has no significant skew towards either gender. As such, the number of hires \
                       is more or less equal among the two genders.')
        with algogen:
            self.__bar_graph(y_predict, unbiased_dataframe, "Algorithm's Hiring Decisions Given The Unbiased Data")
            st.info(f'Plotted above are the hiring decisions made by an algorithm (decision tree classifier) \
                       that was trained off of the biased data. The biased data made decisions on whether or not \
                       to hire an applicant solely off ability, however, when trained on this data, the algorithm \
                       does not follow suit. Despite there not being a purposeful bias towards either gender in the \
                       biased dataset, the algorithm picked up on the skew present and when given unbiased data \
                       (as shown above), showed a significant skew towards the {self.bias} gender.')
        with algo:
            self.__show_classifier(clf)

if __name__ == "__main__":
    ui = UI()
    data = ui.run_generator()
    if data:
        clf_vars, clf = ui.run_classifier(data)
        y_predict, X_test, y_test, y_train = clf_vars
        ui.display_analytics(y_train, y_test, y_predict, data, clf)
