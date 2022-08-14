import pandas as pd
import numpy as np
import random
import statistics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import warnings
warnings.filterwarnings("ignore")

''' Adjusts feature values for the purpose of assessing directionality '''
def change_feature_values(attributes, subject_count=60, feature_index=None, change_amount=-2):
    for i in range(subject_count):
        if feature_index is not None:
            attributes.iloc[i, feature_index] += change_amount
    return attributes


def NN_model(attributes, classifier, holdout_attributes, holdout_classifiers, input_dim=191, subject_count=60, artificial_count=0):
    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(10, kernel_initializer="normal"))
    model.add(layers.Dense(100, activation='relu'))

    model.summary()

    model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer="adam",
    metrics=[keras.metrics.RootMeanSquaredError()],
    )

    # model.fit(attributes, classifier, validation_data=(attributes, classifier), batch_size=pd.DataFrame(attributes).max(), epochs=1)
    model.fit(np.asarray(attributes).astype('float32'), np.asarray(classifier).astype('float32'), validation_data=(np.asarray(holdout_attributes).astype('float32'), np.asarray(holdout_classifiers).astype('float32')), batch_size=64, epochs=200)
    # model.fit(attributes, classifier, validation_data=(attributes, classifier), batch_size=64, epochs=200)
    return model

doing_weights = False

# Manipulate table
file1 = "Modified_Data_New.csv"
file2 = "primary_functions.csv"
features = pd.read_csv(file1)
# print(features)
features = features.set_index("Subject")

# Fill gaps in features
feature_columns = list(features.columns.values)
for feature in feature_columns:
    mean = features[feature].mean()
    features[feature].fillna(mean, inplace = True)

classifiers = pd.read_csv(file2)
classifiers = classifiers.set_index("Category")
classifier_columns = list(classifiers.columns.values)

permuting = False
feature_count = 30

scores = []
feature_weight_table = pd.DataFrame(index=features.columns.values)

''' Go through classifiers and train and test models on them '''
# for i in range(1, len(classifier_columns)):
for i in range(6,7):

    if classifier_columns[i] == "ADCP Teg1" or classifier_columns[i] =="ADCP Teg2":
        continue

    # Generate predictive model for each classifier
    column = classifier_columns[i]
    new_table = pd.DataFrame()
    new_classifier = []

    # Remove subjects from consideration that lack an entry for the classifier
    for j in range(len(features.index) - 1, -1, -1):

        # Find the subject in feature data
        name = list(features.index)
        name = name[j]
        if name[0] == "p":
            if len(name) > 5:
                if name[5] == "-":
                    # Find subject in functional data
                    for k in range(len(classifiers.index)):
                        name_0 = list(classifiers.index)[k]

                        if name_0[0] == "p":
                            if name_0[1:8] == name[2:9] and not np.isnan(classifiers[column].iloc[k]):
                                new_table = new_table.append(features.iloc[j])
                                new_classifier.append(classifiers[column].iloc[k])
                                break
            
            else:
                # Find subject in functional data
                for k in range(len(classifiers.index)):
                    name_0 = list(classifiers.index)[k]
                    if name_0[0] == "p":
                        if name_0[1:4] == name[2:5] and not np.isnan(classifiers[column].iloc[k]):
                            new_table = new_table.append(features.iloc[j])
                            new_classifier.append(classifiers[column].iloc[k])
                            break

        elif name[:2] == "np":
            for k in range(len(classifiers.index)):
                name_0 = list(classifiers.index)[k]
                if name_0[2:5] == name[3:6] and not np.isnan(classifiers[column].iloc[k]):
                    new_table = new_table.append(features.iloc[j])
                    new_classifier.append(classifiers[column].iloc[k])
                    break

    # new_table.apply(zscore)
    # Normalize attributes
    if False:
        averages = new_table.mean(axis=0)
        standard_deviations = new_table.std(axis=0)
        for k in range(len(new_table.index)):
            for l in range(len(new_table.columns.values)):
                new_table.iat[k, l] = (new_table.iat[k, l] - averages[l]) / standard_deviations[l]
    new_classifier = np.asarray(new_classifier)
    
    if permuting:
        new_classifier_copy = new_classifier.copy()
        random.seed(None)
        for n in range(len(new_classifier)):
            # new_classifier[n] = random.uniform(min(new_classifier_copy), max(new_classifier_copy))
            new_classifier[n] = random.gauss(statistics.mean(new_classifier_copy), statistics.pstdev(new_classifier_copy))

    # new_classifier = zscore(new_classifier)
    # Build the model
    file_name = str(classifier_columns[i] + "_predictions.csv")
    out_file = open(file_name, "w")
    for n in range(1):
        x_train, x_test, y_train, y_test = train_test_split(new_table, new_classifier, test_size=0.2, random_state=n)
        # estimator = LinearRegression()
        # estimator = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet'))
        # estimator.fit(x_train.values, y_train)
        # score = estimator.score(x_test, y_test)
        # predictions = new_estimator.predict(x_test.values)
        # predictions = estimator.predict(x_test.values)
        # score = mean_squared_error(y_test, predictions)

        # for p in range(len(predictions)):
        #     out_file.write(str(predictions[p]) + "," + str(y_test[p]) + "\n")

        # print(len(new_table.columns.values))
        model = NN_model(x_train, y_train, x_test, y_test, input_dim=len(new_table.columns.values))
        # predictions = []
        # for row in x_test.values:
        #     print("ROW: " + str(row))
        #     # to_predict = x_test[row]
        #     prediction = model.predict(np.asarray(row).astype('float32'))
        #     predictions.append(prediction)

        # estimator = Ridge(alpha=0.1, solver='saga')
        # feature_count = 15
        # sfs = SFS(estimator, k_features=feature_count, forward=True, floating=False, verbose=2,scoring='neg_mean_squared_error', cv=10)
        # # sfs.fit(new_table, new_classifier)
        # sfs.fit(x_train, y_train)
        # reduced_x_train = sfs.transform(x_train)
        # reduced_x_test = sfs.transform(x_test)
        # estimator = Ridge(alpha=0.1, solver='saga')
        # estimator.fit(reduced_x_train, y_train)
        # predictions = estimator.predict(reduced_x_test)
        # print("\nPREDICTIONS")
        # print(predictions)
        # for z in range(len(predictions)):
        #     print(predictions[z])
        # print("\n\n\ny_test")
        # for item in y_test:
        #     print(item)
        # score = mean_squared_error(y_test, predictions)
        # print("\nSCOREEEEE")
        # print(score)
        # reduced_features = sfs.transform(new_table)
        # new_estimator = Ridge(alpha=0.1, solver='saga')
        # new_estimator.fit(reduced_features)

    # print(str(column) + "," + str(score))
    if doing_weights:

        ''' Identifying directionality and weights of features '''
        feature_weight_dict = {}
        feature_weight_dict["Base Model"] = 0
        for k in range(len(new_table.index)):
            row = [new_table.iloc[k]]
            # print(row)
            prediction = estimator.predict(row).item()
            # print(prediction)
            feature_weight_dict["Base Model"] += prediction / 60

        names = new_table.columns.values

        for k in range(len(names)):
            current_feature = names[k]
            feature_weight_dict[current_feature] = 0
            features_copy = new_table.copy()
            # print(features_copy)
            features_copy = change_feature_values(features_copy, subject_count=len(new_table.index), feature_index=k, change_amount=-5000)
            # print(features_copy)
            # features_copy = np.delete(features_copy, i, 1)
            # estimator.fit(reduced_features, classifier)
            for l in range(len(new_table.index)):
                row = [features_copy.iloc[l, :]]
                # print(row)
                prediction = estimator.predict(row).item()
                feature_weight_dict[current_feature] += prediction / 60


        # print("\n\nGetting weights, so to speak")
        # print(feature_weight_dict)

        base_val = feature_weight_dict["Base Model"]
        feature_weight_dict_copy = {}
        feature_weight_dict_copy["Base Model"] = base_val

        for k in range(len(names)):
            current_feature = names[k]
            feature_weight_dict_copy[current_feature] = 0
            features_copy = new_table.copy()
            features_copy = change_feature_values(features_copy, subject_count=len(new_table.index), feature_index=k, change_amount=5000)
            # features_copy = np.delete(features_copy, i, 1)
            # estimator.fit(features, classifier)
            for l in range(len(new_table.index)):
                row = [features_copy.iloc[l, :]]
                prediction = estimator.predict(row).item()
                feature_weight_dict_copy[current_feature] += prediction / 60

        # print("\n\nGetting weights, so to speak")
        # print(feature_weight_dict_copy)
        
        # Update table of weights
        new_column = []
        feature_weight_table[classifier_columns[i]] = ""
        for key in feature_weight_dict:
            difference = feature_weight_dict_copy[key] - feature_weight_dict[key]
            new_column.append((key, difference))
        
        for pair in new_column:
            feature_weight_table.at[pair[0], classifier_columns[i]] = pair[1]
    # scores.append(score)

if doing_weights:
    print(feature_weight_table)
    feature_weight_table.to_csv('output/output1.csv')
print(','.join(str(e) for e in classifier_columns))
# print(','.join(str(e) for e in scores))