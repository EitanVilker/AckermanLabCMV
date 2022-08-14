import numpy as np
import pandas as pd
from scipy.stats import linregress

''' Check if string can be converted to float '''
def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def POMS(the_list):
    if len(the_list) == 0:
        return the_list
    a = max(the_list)
    b = min(the_list)
    if a != b:
        for i in range(len(the_list)):
            the_list[i] = (the_list[i] - b) / (a - b)
    return the_list

filename = "CMV_Kinetics_Longitudinal.csv"
out_file = "output/output0.csv"

data = open(filename)
lines = data.readlines()

feature_names = ["ADCP gB", "ADCP Pentamer", "ADCP_CG1 Nexelis", "ADCP_CG2 Nexelis", "ADNP gB", "ADNP Pentamer", "ADCD gB", "ADCD Pentamer", "ADCD Teg1", "ADCD Teg2", "ADCD TT"]

# Set up a dictionary of features to the total value and count for each one so as to get the average
# feature_average_dict = {}
# line_number = 0
# for line in lines:
#     if line_number > 1:
#         for i in range(len(feature_names)):
#             current_feature = feature_names[i]
#             feature_average_dict[current_feature] = (0, 0)
#             for j in range(5):
#                 tup = feature_average_dict[current_feature]
#                 entry = line[i * 5 + j + 2]
#                 if entry != "":
#                     new_tup = (tup[0] + float(entry), tup[1] + 1)
#                     feature_average_dict[current_feature] = new_tup

df = pd.DataFrame()
line_number = 0

# For each subject
for line in lines:
    line = line.strip(",")
    line = line.strip("\n")
    line = line.split(",")
    # print(line)
    row = []
    if line_number > 1:

        # For each feature
        for i in range(len(feature_names)):
            current_feature = feature_names[i]
            a = []
            b = []
            # For each time point between V1 and V5
            for j in range(5):
                index = i * 5 + j + 1
                # if line_number == 
                value = line[index]

                if is_float(value):
                    a.append(j + 1)
                    b.append(float(value))

            # b = POMS(b)
            if len(a) > 0:
                slope = linregress(a, b).slope
            else:
                slope = 0

            # Add first subject and column names
            if line_number == 2:
                df[current_feature] = 0
            row.append(slope)

        df.loc[line[0]] = row
        
        # df.rename(index={line_number - 1: line[0]})
    line_number += 1

data.close()

df.to_csv(out_file)