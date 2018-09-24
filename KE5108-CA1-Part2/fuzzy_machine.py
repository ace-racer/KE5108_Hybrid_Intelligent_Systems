import numpy as np
import skfuzzy.control as ctrl
import skfuzzy as fuzz
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class AccountFactors:
    max_activity = 15000.0
    max_balance = 150000.0

    def __init__(self):
        # activity fuzzy
        activity = ctrl.Antecedent(np.arange(0, self.max_activity, 1), 'activity')
        investment_score = ctrl.Consequent(np.arange(0, 11, 1), 'investment_score')
        activity.automf(3)
        investment_score.automf(3)
        rule1 = ctrl.Rule(activity['poor'], investment_score['poor'])
        rule2 = ctrl.Rule(activity['average'], investment_score['average'])
        rule3 = ctrl.Rule(activity['good'], investment_score['good'])
        self.activity_is_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.activity_is_t = ctrl.ControlSystemSimulation(self.activity_is_ctrl)

        # balance fuzzy
        balance = ctrl.Antecedent(np.arange(0, self.max_balance, 1), 'balance')
        investment_score = ctrl.Consequent(np.arange(0, 11, 1), 'investment_score')
        balance.automf(3)
        investment_score.automf(3)
        rule1 = ctrl.Rule(balance['poor'], investment_score['poor'])
        rule2 = ctrl.Rule(balance['average'], investment_score['average'])
        rule3 = ctrl.Rule(balance['good'], investment_score['good'])
        self.balance_is_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.balance_is_t = ctrl.ControlSystemSimulation(self.balance_is_ctrl)

    def activity_to_investment_fuzzy(self, activity):
        self.activity_is_t.input['activity'] = activity
        self.activity_is_t.compute()
        return self.activity_is_t.output['investment_score']

    def balance_to_investment_fuzzy(self, balance):
        self.balance_is_t.input['balance'] = balance
        self.balance_is_t.compute()
        return self.balance_is_t.output['investment_score']

    def calculate(self, row):
        act_score = self.activity_to_investment_fuzzy(row['avtrans'])
        bal_score = self.balance_to_investment_fuzzy(row['avbal'])
        return (act_score + bal_score) * 0.5


class PersonalFactors:
    max_income = 25000
    max_age = 95

    def __init__(self):
        # income fuzzy
        income = ctrl.Antecedent(np.arange(0, self.max_income, 1), 'income')
        investment_score = ctrl.Consequent(np.arange(0, 11, 1), 'investment_score')
        income.automf(3)
        investment_score.automf(3)
        rule1 = ctrl.Rule(income['poor'], investment_score['poor'])
        rule2 = ctrl.Rule(income['average'], investment_score['average'])
        rule3 = ctrl.Rule(income['good'], investment_score['good'])
        self.income_is_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.income_is_t = ctrl.ControlSystemSimulation(self.income_is_ctrl)

        # age fuzzy
        age = ctrl.Antecedent(np.arange(0, self.max_age, 1), 'age')
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
        age = ctrl.Antecedent(np.arange(0, self.max_age, 1), 'age')
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
        if sex == 'M':
            mf_score = 6
        elif sex == 'F' and mstatus != 'married':
            mf_score = 5
        else:  # married females
            mf_score = 2

        return mf_score

    def occupation_score(self, occupation):
        if occupation == 'retired':
            occ_score = 1
        elif occupation == 'manuf' or occupation == 'construct':
            occ_score = 2
        elif occupation == 'government':
            occ_score = 3
        else:
            occ_score = 6  # all other professions

        return occ_score

    def education_score(self, education):
        if education == 'postgrad' or education == 'professional':
            return 6
        elif education == 'tertiary':
            return 4
        elif education == 'secondary':
            return 1

    def calculate(self, row):
        mf_score = self.male_female_score(row['sex'], row['mstatus'])
        occ_score = self.occupation_score(row['occupation'])
        age_score = self.age_to_investment_fuzzy(row['age'])
        income_score = self.income_to_investment_fuzzy(row['income'])
        education_score = self.education_score(row['education'])
        income_edu_score = self.adjust_income_education_to_age(income_score, education_score, row['age'])

        return (mf_score + occ_score + age_score + income_edu_score) / 4.0


af = AccountFactors()
pf = PersonalFactors()

# testrow = {'sex': 'M', 'mstatus': 'single', 'occupation': 'retired', 'age': 45, 'income': 20000,
#            'education': 'tertiary', 'avbal': 20000, 'avtrans': 3000}
# afscore = af.calculate(testrow)
# pfscore = pf.calculate(testrow)
#
# # Use weighting of account, personal {-0.5, 0.5}
# wt = -0.4
# final_score = ((1 - wt) * afscore + (1 + wt) * pfscore) * 0.5


df = pd.read_csv('./original_data/custdatabase.csv')
df = df.rename(columns=lambda x: x.strip())
#
# for wt in [-0.5, -0.45, -0.4, -0.35]:
wt = -0.45
for index, row in df.iterrows():
    afscore = af.calculate(row)
    pfscore = pf.calculate(row)
    # wt = -0.45  # weigh heavily towards account factors : full range (-0.5, 0.5)
    final_score = ((1 - wt) * afscore + (1 + wt) * pfscore) * 0.5
    df.loc[index, 'score'] = round(final_score, 2)

df.to_csv('./results/predicted_scores.csv', index=False)

# df_trial = pd.read_csv('./original_data/trialPromoResults.csv')
# df_trial = df_trial.rename(columns=lambda x: x.strip())
# df = df_trial
# wt = -0.45
# for index, row in df.iterrows():
#     afscore = af.calculate(row)
#     pfscore = pf.calculate(row)
#     # wt = -0.45  # weigh heavily towards account factors : full range (-0.5, 0.5)
#     final_score = ((1 - wt) * afscore + (1 + wt) * pfscore) * 0.5
#     df.loc[index, 'score'] = round(final_score, 2)
#
# df.drop(df.columns[0],axis=1,inplace=True)
#
#
# df.to_csv('./results/trialPromoResults_withIPscores.csv', index=False)
