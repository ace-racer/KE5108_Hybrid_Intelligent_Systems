import numpy as np
import skfuzzy.control as ctrl
import skfuzzy as fuzz
import pandas as pd


class PersonalFactors:

    def __init__(self):
        # income fuzzy
        income = ctrl.Antecedent(np.arange(0, 11, 1), 'income')  # TODO replace 11 by max income
        investment_score = ctrl.Consequent(np.arange(0, 11, 1), 'investment_score')
        income.automf(3)
        investment_score.automf(3)
        rule1 = ctrl.Rule(income['poor'], investment_score['poor'])
        rule2 = ctrl.Rule(income['average'], investment_score['average'])
        rule3 = ctrl.Rule(income['good'], investment_score['good'])
        self.income_is_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.income_is_t = ctrl.ControlSystemSimulation(self.income_is_ctrl)

        # age fuzzy
        age = ctrl.Antecedent(np.arange(0, 11, 1), 'age')  # TODO Replace 11 by max age
        investment_score = ctrl.Consequent(np.arange(0, 11, 1), 'investment_score')
        age.automf(5, names=['young', 'teen', 'middle-aged', 'senior', 'old'])
        investment_score.automf(3)
        rule1 = ctrl.Rule(age['young'], investment_score['poor'])
        rule2 = ctrl.Rule(age['teen'], investment_score['average'])
        rule3 = ctrl.Rule(age['middle-aged'], investment_score['good'])
        rule4 = ctrl.Rule(age['senior'], investment_score['average'])
        rule5 = ctrl.Rule(age['old'], investment_score['poor'])
        self.age_is_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.age_is_t = ctrl.ControlSystemSimulation(self.age_is_ctrl)

        # income - education weighting
        age = ctrl.Antecedent(np.arange(0, 11, 1), 'age')
        ie_weight = ctrl.Consequent(np.arange(-0.5, 0.5, 0.1), 'income_education weight')
        age.automf(5, names=['young', 'teen', 'middle-aged', 'senior', 'old'])
        ie_weight.automf(5)
        rule1 = ctrl.Rule(age['middle-aged'], ie_weight['good'])
        rule2 = ctrl.Rule(age['senior'], ie_weight['mediocre'])
        rule3 = ctrl.Rule(age['old'], ie_weight['poor'])  # poor here means weighted towards income side
        rule4 = ctrl.Rule(age['young'] | age['teen'], ie_weight['average'])
        self.ie_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.ie_t = ctrl.ControlSystemSimulation(self.ie_ctrl)

    def income_to_investment_fuzzy(self, income):
        self.income_is_t.input['income'] = income
        self.income_is_t.compute()
        return self.income_is_t.output['investment_score']

    def age_to_investment_fuzzy(self, age):
        self.age_is_t.input['age'] = age
        self.age_is_t.compute()
        return self.age_is_t.output['investment_score']

    def age_to_ie_weight(self, age):
        self.ie_t.input['age'] = age
        self.ie_t.compute()
        return self.ie_t.output['income_education weight']

    # returns one combined score
    def adjust_income_education_to_age(self, income, education, age):
        wt = self.age_to_ie_weight(age)
        return ((1 - wt) * income + (1 + wt) * education) * 0.5

    def male_female_score(self, sex, mstatus):
        mf_score = 0
        if sex == 'M':
            mf_score = 8
        elif sex == 'F' and mstatus != 'married':
            mf_score = 7
        else:  # all other females
            mf_score = 6

        return mf_score

    def occupation_score(self, occupation):
        if occupation == 'retired':
            occ_score = 2
        else:
            occ_score = 8  # all other professions

        return occ_score

    def education_score(self, education):
        if education == 'postgrad' or education == 'professional':
            return 8
        elif education == 'tertiary':
            return 6
        elif education == 'secondary':
            return 4

    def calculate(self, row):
        mf_score = self.male_female_score(row['sex'], row['mstatus'])
        occ_score = self.occupation_score(row['occupation'])
        age_score = self.age_to_investment_fuzzy(row['age'])
        income_score = self.income_to_investment_fuzzy(row['income'])
        education_score = self.education_score(row['education'])
        income_edu_score = self.adjust_income_education_to_age(income_score, education_score, row['age'])

        return (mf_score + occ_score + age_score + income_edu_score) / 4.0


pf = PersonalFactors()

# Testing
# income_score = pf.income_to_investment_fuzzy(6.5)
# age_score = pf.age_to_investment_fuzzy(5)
# ie_weight = pf.age_to_ie_weight(0)
# mf_score = pf.male_female_score('M', 'married')
# occ_score = pf.occupation_score('retired')
# edu_score = pf.education_score('tertiary')
# income_edu_score = pf.adjust_income_education_to_age(8, 5, 0)  # age 10 == old guy

testrow = {'sex': 'F', 'mstatus': 'single', 'occupation': 'retired', 'age': 0, 'income': 2,
           'education': 'tertiary'}
print(testrow)
pf.calculate(testrow)

# print(income_score, age_score, ie_weight, mf_score, occ_score, edu_score)
# trial_promo_results = pd.read_csv('./original_data/trialPromoResults.csv')
