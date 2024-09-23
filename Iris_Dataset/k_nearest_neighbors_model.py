from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#Load the Iris dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target
labels = iris_data.target_names
features = iris_data.feature_names

#Split the data into training and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Variables to track the best model
best_k = None
best_accuracy = 0
best_model = None

# Loop through different k values
for k in range(1, 21):
    
    # Create and train the model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Update the best model if current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        best_model = knn

# Final output for the best k
print(f"Best k: {best_k} with accuracy: {best_accuracy:.2f}")

# Make predictions
y_pred = best_model.predict(X_test)

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for Best Model:\n", conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report for Best Model:\n", class_report)

# Accuracy score
print(f"Accuracy of Best Model: {best_accuracy:.2f}")



