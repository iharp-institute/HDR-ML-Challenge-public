import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Directory to read labels from
input_dir = sys.argv[1]
solutions = os.path.join(input_dir, 'ref')
prediction_dir = os.path.join(input_dir, 'res')

# Directory to output computed score into
output_dir = sys.argv[2]

def read_prediction():
    prediction_file = os.path.join(prediction_dir,'test.predictions')

    # Check if file exists
    if not os.path.isfile(prediction_file):
        print('[-] Test prediction file not found!')
        print(prediction_file)
        return


    f = open(prediction_file, "r")
    predicted_scores = [list(map(int, line.split())) for line in f.readlines()]
    print(predicted_scores)
    return predicted_scores


def get_t_and_anomaly_data(solution_file):
    # Check if file exists
    if not os.path.isfile(solution_file):
        print('[-] Test solution file not found!')
        return

    # Read the solution file
    df = pd.read_csv(solution_file)
    return df['anomaly']

def read_solution():

    print(os.listdir(solutions))

    # Fix Me 
    # This does not work yet 
    # Load all locations files 
    # Order of the files matter
    list_of_locations = [
        'Atlantic_City_2008_2013_answer_data.csv', 
        'Baltimore_2008_2013_answer_data.csv', 
        'Eastport_2008_2013_answer_data.csv', 
        'Fort_Pulaski_2008_2013_answer_data.csv', 
        'Lewes_2008_2013_answer_data.csv', 
        'New_London_2008_2013_answer_data.csv', 
        'Newport_2008_2013_answer_data.csv', 
        'Portland_2008_2013_answer_data.csv', 
        'Sandy_Hook_2008_2013_answer_data.csv', 
        'Sewells_Point_2008_2013_answer_data.csv', 
        'The_Battery_2008_2013_answer_data.csv', 
        'Washington_2008_2013_answer_data.csv' 
    ]


    test_answer_location = [get_t_and_anomaly_data(os.path.join(solutions, location)).size for location in list_of_locations] 

    print(test_answer_location)

    return test_answer_location


def save_score(f1_t_only, f1_t_and_location, final_grade):
    score_file = os.path.join(output_dir, 'scores.json')

    scores = {
        'F1-Score': f1_t_only,
        'F1-Score with location': f1_t_and_location,
        'Final Grade': final_grade
    }
    with open(score_file, 'w') as f_score:
        f_score.write(json.dumps(scores))
        f_score.close()


def print_pretty(text):
    print("-------------------")
    print("#---",text)
    print("-------------------")


    
def main():

    # Read prediction and solution
    print_pretty('Reading prediction')
    prediction = read_prediction()
    solution = read_solution()

    print(prediction)

    # Compute Score
    print_pretty('Computing score')
    # Fix Me 
    f1_t_only = f1_score(merged_t['anomaly_true'], merged_t['anomaly_pred'])

    # F1-score based on both 't' and 'location' columns
    f1_t_and_location = f1_score(merged_df['anomaly_true'], merged_df['anomaly_pred'])

    # Calculate the final grade with weightages
    final_grade = (0.3 * f1_t_only) + (0.7 * f1_t_and_location)

    # Print the results
    print(f"F1-Score based only on 't': {f1_t_only:.4f}")
    print(f"F1-Score based on 't' and 'location': {f1_t_and_location:.4f}")
    print(f"Final Grade: {final_grade:.4f}")

    # Write Score
    print_pretty('Saving prediction')
    save_score(score)

if __name__ == '__main__':
    main()
