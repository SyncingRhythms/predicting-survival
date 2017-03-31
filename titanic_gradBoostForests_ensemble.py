import pandas as pd
import numpy as np
import re, operator, annotate
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

if __name__ == "__main__":
	#kaggle training set
	titanic = pd.read_csv("train.csv")
	print(titanic.head())

	titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
	print(titanic.describe())

	#convert string columns to numeric
	#print(titanic["Sex"].unique())
	titanic.loc[titanic["Sex"]=="male", "Sex"] = 0
	titanic.loc[titanic["Sex"]=="female", "Sex"] = 1
	titanic.loc[titanic["Embarked"].isnull(), "Embarked"] = "S"
	titanic.loc[titanic["Embarked"]=="S", "Embarked"] = 0
	titanic.loc[titanic["Embarked"]=="C", "Embarked"] = 1
	titanic.loc[titanic["Embarked"]=="Q", "Embarked"] = 2
	titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
	titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

	# Get all the titles and print how often each one occurs.
	titles = titanic["Name"].apply(annotate.get_title)
	print(pd.value_counts(titles))

	# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
	for k,v in title_mapping.items():
		titles[titles == k] = v

	# Verify that we converted everything.
	print(pd.value_counts(titles))

	# Add in the title column.
	titanic["Title"] = titles

	# Get the family ids with the apply method
	family_ids = titanic.apply(annotate.get_family_id, axis=1)

	# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
	family_ids[titanic["FamilySize"] < 3] = -1

	# Print the count of each unique id.
	print(pd.value_counts(family_ids))

	titanic["FamilyId"] = family_ids

	# save cleaned dataset to csv
	titanic.to_csv("titanic_clean.csv")

	###Gradient Boosting###
	## & Random Forest
	# The algorithms we want to ensemble.
	# Along with the feature predictors
	algorithms = [
		[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
		[RandomForestClassifier(random_state=1, min_samples_split=8, min_samples_leaf=4, n_estimators=50), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
	]

	# Initialize the cross validation folds
	kf = KFold(n_splits=10, random_state=3)

	predictions = []
	for train, test in kf.split(titanic):
		train_target = titanic["Survived"].iloc[train]
		full_test_predictions = []
		# Make predictions for each algorithm on each fold
		for alg, predictors in algorithms:
		    # Fit the algorithm on the training data.
		    alg.fit(titanic[predictors].iloc[train,:], train_target)
		    # Select and predict on the test fold.
		    # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
		    test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
		    full_test_predictions.append(test_predictions)
		# Use a simple ensembling scheme -- just average the predictions to get the final classification.
		test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
		# Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
		test_predictions[test_predictions <= .5] = 0
		test_predictions[test_predictions > .5] = 1
		predictions.append(test_predictions)

	# Put all the predictions together into one array.
	predictions = np.concatenate(predictions, axis=0)

	# Compute accuracy by comparing to the training data.
	accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
	print("{0}% of Predictions Correct".format(round(accuracy*100)))
