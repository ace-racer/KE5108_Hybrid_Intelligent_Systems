### The developed scripts are described here

#### Scripts for overall, model and expert system evaluation
1. `evaluate_campaign.py`: Usage: `python evaluate_campaign.py \"H:\\KE 5108 - CAs\\code\\KE 5108 -CA1-Part 2\\original_data\\cust_actual_merged.csv\" cust_actual_merged_best` will sort the input data with the status and the score columns, generated from the classification model and expert system respectively and select the top 400 customers to be targeted for the campaign.

2. `compare_results.py`: Usage:`python compare_results.py "H:\\KE 5108 - CAs\\code\\KE 5108 -CA1-Part 2\\working_data\\trial_output.csv"` will compare the status and the scores from the input file and generate a confusion matrix and mean absolute error respectively. It also generates a CSV file inside `results` folder that has the difference for each record in the input.

#### Notebook to generate test predictions (on the list of 4000 records)
1. The notebook `Generate_Test_Predictions.ipynb` will generate the predictions for the product to be purchased by the 4000 customers. The `model_to_use` parameter in the second cell needs to be provided which is the H5 file storing the trained model.

