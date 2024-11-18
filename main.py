import pandas as pd
import numpy as np
import tensorflow as ts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Load dataset
data = pd.read_csv('C:/Users/user/Desktop/CSV/kidney_disease.csv')

# Check column names
print("Columns in the dataset:", data.columns)

# Display the first few rows of the dataset
print(data.head())

# Assuming the correct column names are 'symptoms', 'condition', and 'solution'
# If the column names are different, adjust them accordingly
try:
    symptoms = data['symptoms']
    conditions = data['condition']
    solutions = data['solution']
except KeyError as e:
    print(f"Error: Column not found - {e}")
    exit()

# Encode symptoms and conditions
le_conditions = LabelEncoder()
le_solutions = LabelEncoder()

conditions_encoded = le_conditions.fit_transform(conditions)
solutions_encoded = le_solutions.fit_transform(solutions)

# Convert symptoms into feature vectors (example using simple bag-of-words)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
symptoms_encoded = vectorizer.fit_transform(symptoms).toarray()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    symptoms_encoded, conditions_encoded, test_size=0.2, random_state=42
)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(conditions_encoded)), activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Predict and suggest solutions
def diagnose(symptom_text):
    symptom_vector = vectorizer.transform([symptom_text]).toarray()
    predicted_condition = np.argmax(model.predict(symptom_vector))
    solution = le_solutions.inverse_transform([predicted_condition])
    return le_conditions.inverse_transform([predicted_condition])[0], solution[0]

# Example usage
symptom_example = "fever, cough, headache"
condition, solution = diagnose(symptom_example)
print(f"Diagnosed Condition: {condition}, Suggested Solution: {solution}")
