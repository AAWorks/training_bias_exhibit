from random import randint, choice
import pandas as pd
import streamlit as st
from scripts.classifier import AlgoUtils

class TestSet:
    """generate the unbiased and biased dataset based on user specficiations"""

    def __init__(self, num_applicants, num_hires, bias):
        self.biased_data = pd.DataFrame(columns=['Gender', 'Skill', 'Academics', 'Experience', 'Ambition'])
        self.unbiased_data = pd.DataFrame(columns=['Gender', 'Skill', 'Academics', 'Experience', 'Ambition'])
        self.total_applicants = num_applicants
        self.hires = num_hires
        self.bias = bias.lower()

    def __get_stats(self, gender, a, b):
        stats = [gender] + [randint(a, b) for _ in range(4)] # 0 is male, 1 is female
        return stats
    
    def __get_cv(self):
        if self.bias == "male":
            if randint(1, 100) < 50:
                cv = self.__get_stats(0, 2, 9)
            else:
                cv = self.__get_stats(1, 2, 6)
        else:
            if randint(1, 100) < 50:
                cv = self.__get_stats(1, 2, 9)
            else:
                cv = self.__get_stats(0, 2, 6)
        return cv

    def biased_dataset(self):
        biased_data = self.biased_data
        progress_text = "Generating biased dataset..."
        my_bar = st.progress(0, text=progress_text)
        unit = self.total_applicants // 100

        for i in range(self.total_applicants):
            cv = self.__get_cv()
            biased_data = biased_data.append({'Gender': cv[0], 'Skill': cv[1], 'Academics': cv[2], 'Experience': cv[3], 'Ambition': cv[4]}, ignore_index = True)
            if i % unit == 0:
                percent_complete = i / unit
                my_bar.progress(percent_complete / 100, text=progress_text)

        target = AlgoUtils.label_accepted(biased_data, self.hires)
        my_bar.progress(1.0, text="Generated Biased Dataset :white_check_mark:")
        return biased_data, (biased_data.values, target)

    def unbiased_dataset(self):
        unbiased_data = self.unbiased_data
        progress_text = "Generating unbiased dataset..."
        my_bar = st.progress(0, text=progress_text)
        unit = self.total_applicants // 100

        for i in range(self.total_applicants):
            unbiased_data = unbiased_data.append({'Gender': choice([0, 1]), 'Skill': randint(2, 9), 'Academics': randint(2,9), 'Experience': randint(2,9), 'Ambition': randint(2,9)}, ignore_index = True)
            if i % unit == 0:
                percent_complete = i / unit
                my_bar.progress(percent_complete / 100, text=progress_text)

        target = AlgoUtils.label_accepted(unbiased_data, self.hires)
        my_bar.progress(1.0, text="Generated Unbiased Dataset :white_check_mark:")
        return unbiased_data, (unbiased_data.values, target)
