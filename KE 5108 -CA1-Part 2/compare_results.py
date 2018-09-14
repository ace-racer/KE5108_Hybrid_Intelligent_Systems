import pandas as pd
import numpy as np
import sys

NUM_CUSTOMERS = 400

def compare_status_score_with_actuals(actuals_df, current_df):
    if actuals_df is not None and current_df is not None:
        required_customer_ids = current_df["index"]
        actual_status_scores_for_required_customers = actuals_df[actuals_df["index"].isin(required_customer_ids)][["index", "status", "score"]].reset_index(drop=True)
        actual_status_scores_for_required_customers.sort_values(by="index", inplace=True)
        current_df.sort_values(by="index", inplace=True)
        print(actual_status_scores_for_required_customers)
        current_df["actual_status"] = actual_status_scores_for_required_customers["status"]
        current_df["actual_score"] = actual_status_scores_for_required_customers["score"]
        current_df["is_status_diff"] = current_df["actual_status"] != current_df["status"]
        current_df["absolute_score_diff"] = abs(current_df["actual_score"] - current_df["score"])
        print(current_df)

if __name__ == "__main__":
    """python compare_results.py "H:\\KE 5108 - CAs\\code\\KE 5108 -CA1-Part 2\\working_data\\trial_output.csv" """
    if len(sys.argv) != 2:
        print("Usage error")
    else:
        current_df = pd.read_csv(sys.argv[1], header=0, index_col=False)
        actuals_df = pd.read_csv("original_data/cust_actual_merged.csv", header=0, index_col=False)

        if current_df.shape[0] != NUM_CUSTOMERS:
            raise ValueError("Number of customers in the current file should be {0}".format(NUM_CUSTOMERS))

        compare_status_score_with_actuals(actuals_df, current_df)