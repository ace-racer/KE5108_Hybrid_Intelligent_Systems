import pandas as pd
import numpy as np
import os
import sys

NUM_CUSTOMERS_TO_CHOOSE = 400

def get_best_customers_total_score(df, output_name = None):
    """Get the set of best performing customers"""
    if df is not None:
        df["status"] = df["status"].replace("A", 0.6).replace("B", 1).replace("None", 0)
        df["expected_profit"] = df["status"] * df["score"]
        df = df.sort_values("expected_profit", ascending=False)

        if output_name:
            output_file_location = os.path.join("results", output_name + ".csv")
            df.to_csv(output_file_location, index=False)
            print("Generated the output file here: {0}".format(output_file_location))

        total_expected_profit = np.sum(df["expected_profit"][:NUM_CUSTOMERS_TO_CHOOSE], axis=0)
        print("Total expected profit: " + str(total_expected_profit))

def validate_columns(df):
    if df is not None:
        required_columns = ["status", "score"]
        for required_column in required_columns:
            if required_column not in df.columns.values:
                raise ValueError("The required column {0} is not present.".format(required_columns))

# python evaluate_campaign.py "H:\\KE 5108 - CAs\\code\\KE 5108 -CA1-Part 2\\original_data\\cust_actual_merged.csv" cust_actual_merged_best

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_campaign.py \"H:\\KE 5108 - CAs\\code\\KE 5108 -CA1-Part 2\\original_data\\cust_actual_merged.csv\" cust_actual_merged_best")
    else:
        input_file_location = sys.argv[1]

        if len(sys.argv) >= 3:
            output_file_name = sys.argv[2]
        else:
            output_file_name = None
        
        df = pd.read_csv(input_file_location, header=0)
        validate_columns(df)
        get_best_customers_total_score(df, output_file_name)