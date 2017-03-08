import pandas as pd
import numpy as np
import re, operator, annotate
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

if __name__ == "__main__":
	#kaggle training set
	titanic = pd.read_csv("titanic_clean.csv")
	#kaggle submission testing set
	titanic_test = pd.read_csv("test.csv")
	print(titanic_test.head())

	#must fill test values with the same from the training set, unless there's missing values in the test set that weren't in the training set -- like the fare column in the test set, which we will replace with the titanic_test["Fare"].median() (i.e., from the testing dataset)
	titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
	titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
	#print(titanic.describe())

	#convert string columns to numeric
	#print(titanic_test["Sex"].unique())
	titanic_test.loc[titanic_test["Sex"]=="male", "Sex"] = 0
	titanic_test.loc[titanic_test["Sex"]=="female", "Sex"] = 1
	titanic_test.loc[titanic_test["Embarked"].isnull(), "Embarked"] = "S"
	titanic_test.loc[titanic_test["Embarked"]=="S", "Embarked"] = 0
	titanic_test.loc[titanic_test["Embarked"]=="C", "Embarked"] = 1
	titanic_test.loc[titanic_test["Embarked"]=="Q", "Embarked"] = 2
	#print(titanic.Embarked.unique())

	# Map titles
	# First, we'll add titles to the test set.
	titles = titanic_test["Name"].apply(annotate.get_title)
	# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
	for k,v in title_mapping.items():
	    titles[titles == k] = v
	titanic_test["Title"] = titles
	# Check the counts of each unique title.
	print(pd.value_counts(titanic_test["Title"]))

	# Now, we add the family size column to the testing set
	titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

	# add family ids
	# Get the family ids with the apply method
	family_ids = titanic_test.apply(annotate.get_family_id, axis=1)
	family_ids[titanic_test["FamilySize"] < 3] = -1
	titanic_test["FamilyId"] = family_ids

	# lastly, add the name length variable
	titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))


	###GENERATE NOVEL PREDICTIONS###
	## on kaggle test set ##
	predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

	algorithms = [
	    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
	    [RandomForestClassifier(random_state=1, min_samples_split=8, min_samples_leaf=4, n_estimators=50), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
	]

	full_predictions = []
	for alg, predictors in algorithms:
	    # Fit the algorithm using the full training data.
	    alg.fit(titanic[predictors], titanic["Survived"])
	    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
	    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
	    full_predictions.append(predictions)

	# The gradient boosting classifier generates better predictions, so we weight it higher.
	predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
	predictions[predictions > .5] = 1
	predictions[predictions <= .5] = 0
	predictions = predictions.astype(int)

	submission = pd.DataFrame({
	    "PassengerId": titanic_test.PassengerId,
	    "Survived": predictions
	})
	submission.to_csv("titanic_boost_predictions.csv", index=False)

