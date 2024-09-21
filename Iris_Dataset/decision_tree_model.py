import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import tree


# Load the Iris dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target
labels = iris_data.target_names
features = iris_data.feature_names

#Split the data into training and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Define a parameter grid to search for the best parameters
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Use the best model found by grid search
best_model = grid_search.best_estimator_

# Predict with the best model
y_pred = best_model.predict(X_test)

#Evaluate model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_importances = best_model.feature_importances_
features = iris_data.feature_names

#Visualize the decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(best_model, feature_names=features, class_names=labels, filled=True)

#Save the plot to a file
plt.savefig("decision_tree.png")

#Display plot
plt.show()
