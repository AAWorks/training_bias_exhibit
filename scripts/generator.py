from random import randint, choice
import pandas as pd

class Generator:
    """generate the unbiased and biased dataset based on user specficiations"""

    data: pd.DataFrame
    total_applicants: int
    hires: int
    def __init__(self, num_applicants, num_hires, larger_ratio, bias):
        self.data = pd.DataFrame(columns=['Gender', 'Skill', 'Academics', 'Experience', 'Ambition'])
        self.total_applicants = num_applicants
        self.hires = num_hires
        self.larger_ratio = larger_ratio
        self.bias = bias

    def __get_stats(self, gender, a, b):
        stats = [gender] + [randint(a, b) for _ in range(4)] # 0 is male, 1 is female
        return stats
    
    def __get_cv(self, ratios):
        if ratios[1] == 'male':
            if randint(1, 100) < ratios[0]:
                cv = generate_stats(0, 4, 9)
            else:
                cv = generate_stats(1, 2, 6)
        else:
            if randint(1, 100) < ratios[0]:
                cv = generate_stats(1, 4, 9)
            else:
                cv = generate_stats(0, 2, 6)
        return cv

    def biased_dataset(self, biased_data):
        print('Creating the biased dataset...')

        ratios = define_ratios()
        for applicant in range(self.total_applicants):
            cv = generate_cv(ratios)
            biased_data = biased_data.append({'Gender': cv[0], 'Skill': cv[1], 'Academics': cv[2], 'Experience': cv[3], 'Ambition': cv[4]}, ignore_index = True)
        target=functions.label_accepted(biased_data, self.hires)
        print('Done. \n')
        return biased_data, [biased_data.values, target]
  def unbiased_dataset(self, unbiased_data):
    print('Creating unbiased dataset...')
    for applicant in range(self.total_applicants):
      unbiased_data = unbiased_data.append({'Gender': choice([0, 1]), 'Skill': randint(2, 9), 'Academics': randint(2,9), 'Experience': randint(2,9), 'Ambition': randint(2,9)}, ignore_index = True)
    target=functions.label_accepted(unbiased_data, self.hires)
    print('Done. \n')
    return unbiased_data, [unbiased_data.values, target]
