import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.generator import TestSet
from scripts.classifier import AlgoUtils

class UI:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Automated Hiring Simulator", page_icon=":man_in_tuxedo:")
        st.title("Exhibiting Training Bias in Simple Automated Hiring")
    
    def __generator_inputs(self):
        with st.form(key='generator'):
            st.write("Dataset Generator Inputs")
            col0, col1, col2 = st.columns(3)
            applicants = col0.number_input("Number of Applicants", min_value = 100, value=10000, step=100)
            hires = col1.number_input("Number of Hires", max_value=applicants, min_value=50, value=500, step=100)
            bias = col2.selectbox(
                'Data Skew',
                ("Male", "Female"))
            submit = st.form_submit_button(label='Generate', use_container_width=True)
        if submit:
            return (int(applicants), int(hires), bias)

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
        test_vars = AlgoUtils.train(bias_data[0], no_bias_data[0], bias_data[1], no_bias_data[1])
        return AlgoUtils.test(*test_vars)
    
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
    
    def display_analytics(self, y_train, y_test, y_predict, data):
        biased_dataframe, unbiased_dataframe = data[0], data[2]
        train, algogen, pregen = st.tabs(["Pre-Generated Biased Hiring Data", 
                                          'Algorithm Generated Hiring on Unbiased Data',
                                          "Pre-Generated Unbiased Hiring Data"])
        with train:
            self.__bar_graph(y_train, biased_dataframe, "Pre-Generated Biased Hiring Data")
        with algogen:
            self.__bar_graph(y_predict, unbiased_dataframe, 'Algorithm Generated Hiring On Unbiased Data')
        with pregen:
            self.__bar_graph(y_test, unbiased_dataframe, 'Pre-Generated Unbiased Hiring Data')

if __name__ == "__main__":
    ui = UI()
    data = ui.run_generator()
    if data:
        y_predict, X_test, y_test, y_train = ui.run_classifier(data)
        ui.display_analytics(y_train, y_test, y_predict, data)

