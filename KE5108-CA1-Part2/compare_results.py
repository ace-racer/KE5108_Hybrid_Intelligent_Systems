import pandas as pd
import numpy as np
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools

NUM_CUSTOMERS = 400


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def compare_status_score_with_actuals(actuals_df, current_df):
    if actuals_df is not None and current_df is not None:
        required_customer_ids = current_df["index"]

        actual_status_score_df = actuals_df.loc[:, ["index", "status", "score"]]
        actual_status_score_df["actual_status"] = actual_status_score_df["status"]
        actual_status_score_df["actual_score"] = actual_status_score_df["score"]
        actual_status_score_df = actual_status_score_df.loc[:, ["index", "actual_status", "actual_score"]]
    
        current_df = current_df.set_index("index").join(actual_status_score_df.set_index("index"))
        
        current_df["actual_status_multiplier"] = current_df["actual_status"].replace("A", 0.6).replace("B", 1).replace("None", 0)
        current_df["is_status_diff"] = current_df["actual_status"] != current_df["status"]
        current_df["actual_profit"] = current_df["actual_status_multiplier"] * current_df["actual_score"]
        current_df["absolute_score_diff"] = abs(current_df["actual_score"] - current_df["score"])
        
        print("final file")
        print(current_df.head(5))
        
        current_df.to_csv("results/comparison_results.csv", index=False)
        print("Generated the comparison results here: {0}".format("results/comparison_results.csv"))

        mean_absolute_error = np.average(current_df["absolute_score_diff"])
        print("Mean absolute error of scores: {0}".format(mean_absolute_error))

        cm = confusion_matrix(current_df["actual_status"], current_df["status"])
        a = accuracy_score(current_df["actual_status"], current_df["status"])

        current_df.head(5)
        precision, recall, fscore, support = precision_recall_fscore_support(current_df["actual_status"],
                                                                             current_df["status"],
                                                                             labels=["None", "A", "B"])
        score_dict = {
            "precision": precision.round(4),
            "recall": recall.round(4),
            "f1-score": fscore.round(4),
            "support": support
        }
        score_df = pd.DataFrame(score_dict, index=["None", "A", "B"])
        print(score_df)
        print("Accuracy: {0}".format(a))
        # print("F1 score: {0}".format(f1))
        plot_confusion_matrix(cm, classes=["A", "B", "None"], title='Confusion matrix, in numbers')


if __name__ == "__main__":
    """python compare_results.py "H:\\KE 5108 - CAs\\code\\KE 5108 -CA1-Part 2\\results\\cust_actual_merged_best.csv" """
    if len(sys.argv) != 2:
        print("Usage error")
    else:
        print(sys.argv[1])
        current_df = pd.read_csv(sys.argv[1], header=0, index_col=False)
        actuals_df = pd.read_csv("original_data/cust_actual_merged.csv", header=0, index_col=False)

        # if current_df.shape[0] != NUM_CUSTOMERS:
        # raise ValueError("Number of customers in the current file should be {0}".format(NUM_CUSTOMERS))

        compare_status_score_with_actuals(actuals_df, current_df)
