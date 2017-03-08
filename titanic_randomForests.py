import pandas as pd
import numpy as np
import operator, annotate
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,  cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

if __name__ == "__main__":
	#kaggle training set
	titanic = pd.read_csv("train.csv")
	print(titanic.head())

	titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
	print(titanic.describe())

	#convert string columns to numeric
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

	#save cleaned dataset
	titanic.to_csv("titanic_clean.csv")

	##Random Forest###
	###Feature Selection###
	# Perform feature selection
	predictors = list(titanic.columns.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"]))
	selector = SelectKBest(f_classif, k=5)
	selector.fit(titanic[predictors], titanic["Survived"])

	# Get the raw p-values for each feature, and transform from p-values into scores
	scores = -np.log10(selector.pvalues_)

	# Plot the scores.
	plt.bar(range(len(predictors)), scores)
	plt.xticks(range(len(predictors)), predictors, rotation='vertical')
	plt.show()

	# Pick only the four best features.
	predictors = ["Pclass", "Sex", "Fare", "Title"]#"NameLength"
	##cross-validation###
	alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

	kf = KFold(n_splits=3, random_state=1)
	scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
	print("{0}% of Predictions Correct".format(round(scores.mean()*100)))
