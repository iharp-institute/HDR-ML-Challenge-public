import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class Model:
    def __init__(self):
        print("Dummy Model initialized")
        # Put your initialization code here
        # load the save model here
        

    def predict(self, X):

        # Put your prediction code here
        # This example predicts a random value for 12 station
        # The output should be a dataframe with 10 rows and 12 columns
        # Each value should be 1 for anamoly and 0 for normal
        # Return a np array of 1s and 0s with the same length of 12
        # with random prediction of 1 or 0
        return torch.randint(0, 2, (12,)).numpy()
        