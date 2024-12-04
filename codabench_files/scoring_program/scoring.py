import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import csv

# Directory to read labels from
input_dir = sys.argv[1]
solutions = os.path.join(input_dir, "ref")
prediction_dir = os.path.join(input_dir, "res")

# Directory to output computed score into
output_dir = sys.argv[2]


def read_prediction():
    prediction_file = os.path.join(prediction_dir, "demo_sla_submission_format.csv")

    # Check if file exists
    if not os.path.isfile(prediction_file):
        print("[-] Test prediction file not found!")
        print(prediction_file)
        return

    # Read the prediction file into a dataframe
    df_predicted = pd.read_csv(prediction_file)
    return df_predicted


def read_solution():
    solution_file = os.path.join(solutions, "model_solution.csv")

    # Check if file exists
    if not os.path.isfile(solution_file):
        print("[-] Solution file not found!")
        print(solution_file)
        return

    # Read the solution file into a dataframe
    df_solution = pd.read_csv(solution_file)
    return df_solution


def compute_metrics(prediction, solution):
    # Merge the dataframes on the Date column
    merged_df = pd.merge(prediction, solution, on="Date", suffixes=("_pred", "_true"))

    # Initialize lists to store metrics for each location
    tpr_list = []
    fpr_list = []
    f1_list = []

    # Iterate over each location column
    for location in prediction.columns[1:]:  # Skip the 'Date' column
        y_true = merged_df[f"{location}_true"]
        y_pred = merged_df[f"{location}_pred"]

        # Calculate True Positives, False Positives, True Negatives, and False Negatives
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        # Calculate TPR, FPR, and F1-score
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = f1_score(y_true, y_pred)

        # Append the metrics to the lists
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        f1_list.append(f1)

    # Calculate the average metrics across all locations
    avg_tpr = np.mean(tpr_list)
    avg_fpr = np.mean(fpr_list)
    avg_f1 = np.mean(f1_list)

    return avg_tpr, avg_fpr, avg_f1


def save_score(avg_tpr, avg_fpr, avg_f1):
    score_file = os.path.join(output_dir, "scores.json")

    scores = {
        "Average True Positive Rate (TPR)": avg_tpr,
        "Average False Positive Rate (FPR)": avg_fpr,
        "Average F1-Score": avg_f1,
    }
    with open(score_file, "w") as f_score:
        json.dump(scores, f_score, indent=4)


def print_pretty(text):
    print("-------------------")
    print("#---", text)
    print("-------------------")


def main():

    # Read prediction and solution
    print_pretty("Reading prediction")
    prediction = read_prediction()
    solution = read_solution()

    # print(prediction)

    # Compute metrics
    print_pretty("Computing metrics")
    avg_tpr, avg_fpr, avg_f1 = compute_metrics(prediction, solution)

    # Print the results
    print(f"Average True Positive Rate (TPR): {avg_tpr:.4f}")
    print(f"Average False Positive Rate (FPR): {avg_fpr:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    # Save the results
    print_pretty("Saving metrics")
    save_score(avg_tpr, avg_fpr, avg_f1)


if __name__ == "__main__":
    main()
