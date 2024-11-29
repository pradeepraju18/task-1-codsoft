# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Step 2: Explore the dataset (optional)
print("Dataset Overview:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 3: Data Preprocessing
# Fill missing values for 'Age' and 'Embarked'
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop rows where the 'Survived' column is missing (if any)
df.dropna(subset=['Survived'], inplace=True)

# Convert 'Sex' to numeric (0 = male, 1 = female)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numeric (C = 0, Q = 1, S = 2)
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop non-useful columns ('Name', 'Ticket', 'Cabin')
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Step 4: Split the data into features (X) and target (y)
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

# Step 5: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize and train the model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the results
print(f'\nAccuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)


