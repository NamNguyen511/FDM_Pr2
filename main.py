import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score



# Load and prepare the data
images_df = pd.read_csv('Images.csv', delimiter=';', skiprows=1, header=None, names=['ID', 'Class'])
edge_hist_df = pd.read_csv('EdgeHistogram.csv', delimiter=';', skiprows=1, header=None)
edge_hist_df.rename(columns={0: "ID"}, inplace=True)


# Join the datasets on image ID
data = pd.merge(images_df, edge_hist_df, on='ID')
data.drop(columns='ID', inplace=True)


# Split data into features and labels
X = data.iloc[:, 1:]
y = data['Class']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# List to hold confusion matrices, hyperparameters
confusion_matrices = []
hyperparameters_list = []
class_labels = sorted(y.unique(), key=str.lower)

# Classifier configurations
classifiers = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150, 200, 300],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 15, 20, 25],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    },
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'class_weight': ['balanced'],
        }
    },
    'NeuralNetwork': {
        'model': MLPClassifier(random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,100), (100,), (150,), (200,)],
            'activation': ['tanh', 'relu', 'logistic'],
            'batch_size': [20, 40],
            'learning_rate': ['invscaling'],
            'max_iter': [200, 300, 350, 400]
        }
    }
}

# GridSearchCV (trains and evaluates each classifier
for clf_name, clf_conf in classifiers.items():
    grid_search = GridSearchCV(clf_conf['model'], clf_conf['params'], cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    # Store results
    confusion_matrices.append(cm)
    hyperparameters_list.append({
        'name': clf_name,
        'library': 'scikit-learn',
        'test_size': 0.25,
        **grid_search.best_params_
    })

# Export the best hyperparameters and prediction results to csv
def export_to_csv(group_no, result_no, cm, hyperparameters):
    # Confusion Matrix
    cm_filename = f"group{group_no}_result{result_no}.csv"
    with open(cm_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([""] + class_labels)
        for label, row in zip(class_labels, cm):
            writer.writerow([label] + row.tolist())

    # Hyperparameters
    params_filename = f"group{group_no}_parameters{result_no}.csv"
    with open(params_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in hyperparameters.items():
            writer.writerow([key, value])

group_no = '056'
for i, (cm, hyperparameters) in enumerate(zip(confusion_matrices, hyperparameters_list), start=1):
    export_to_csv(group_no, i, cm, hyperparameters)
