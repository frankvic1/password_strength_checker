import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Title for the app
st.title("Password Strength Checker")

# Sample password dataset (for testing)
data = {
    'password': ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!'],
    'length': [len(p) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
    'upper_case': [sum(1 for c in p if c.isupper()) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
    'lower_case': [sum(1 for c in p if c.islower()) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
    'numbers': [sum(1 for c in p if c.isdigit()) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
    'special_chars': [len(re.findall(r'[^a-zA-Z0-9]', p)) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
    'tries': [5, 15000, 10, 8000, 200000, 7000, 180000]
}

# Create DataFrame
pwd_data = pd.DataFrame(data)

# Calculate log2 probability (simplified for now)
pwd_data['log2_prob'] = np.log2(1 / pwd_data['tries'])

# Classify passwords as 'likely' or 'unlikely'
SEPARATOR = -10.0  # Placeholder value
pwd_data['LIKELY'] = (pwd_data['log2_prob'] > SEPARATOR).astype(int)

# Display the dataframe
st.subheader("Sample Password Dataset:")
st.dataframe(pwd_data)

# Visualization
st.subheader("Password Tries vs Log2 Probability")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='log2_prob', y='tries', hue='LIKELY', data=pwd_data)
plt.title('Password Tries vs Log2 Probability')
st.pyplot(plt.gcf())

# Prepare data for models
X = pwd_data[['length', 'upper_case', 'lower_case', 'numbers', 'special_chars', 'log2_prob']]
y = pwd_data['LIKELY']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
st.subheader("Model Evaluation:")
st.write("Decision Tree Accuracy:", metrics.accuracy_score(y_test, dt_pred))

# Logistic Regression
lg_model = LogisticRegression(max_iter=400)
lg_model.fit(X_train, y_train)
lg_pred = lg_model.predict(X_test)
st.write("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, lg_pred))
