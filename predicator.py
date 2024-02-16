import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier

# Load your dataset
df = pd.read_csv("heart_disease.csv")

# Preprocess your data, define features (X) and target variable (y)

# Split the dataset into training and testing sets
x = df.drop(columns=['Disease'])
y = df['Disease']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train your decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Save trained model to a .pkl file
joblib.dump(model, 'decision_tree_model.pkl')

print("Successfully saved the model as decision_tree_model.pkl")
