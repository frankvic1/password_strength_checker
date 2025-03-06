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

# Sample password dataset (for testing)
data = {'password': ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!'],
        'length': [len(p) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
        'upper_case': [sum(1 for c in p if c.isupper()) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
        'lower_case': [sum(1 for c in p if c.islower()) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
        'numbers': [sum(1 for c in p if c.isdigit()) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
        'special_chars': [len(re.findall(r'[^a-zA-Z0-9]', p)) for p in ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']],
        'tries': [5, 15000, 10, 8000, 200000, 7000, 180000]}

pwd_data = pd.DataFrame(data)
pwd_data['log2_prob'] = np.log2(1 / pwd_data['tries'])
pwd_data['LIKELY'] = (pwd_data['log2_prob'] > -10.0).astype(int)

# Prepare data for models
X = pwd_data[['length', 'upper_case', 'lower_case', 'numbers', 'special_chars', 'log2_prob']]
y = pwd_data['LIKELY']

# Train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
lg_model = LogisticRegression(max_iter=400).fit(X_train, y_train)

# Streamlit UI
st.title('Password Strength Checker')
st.subheader('Sample Password Dataset:')
st.dataframe(pwd_data)

st.subheader('Password Tries vs Log2 Probability')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log2_prob', y='tries', hue='LIKELY', data=pwd_data)
st.pyplot(plt)

# Real-time Password Evaluation
st.subheader('Test Your Password Strength')
password = st.text_input('Enter a password:')

if password:
    length = len(password)
    upper_case = sum(1 for c in password if c.isupper())
    lower_case = sum(1 for c in password if c.islower())
    numbers = sum(1 for c in password if c.isdigit())
    special_chars = len(re.findall(r'[^a-zA-Z0-9]', password))
    log2_prob = np.log2(1 / 10000)  # Placeholder tries value

    features = pd.DataFrame([[length, upper_case, lower_case, numbers, special_chars, log2_prob]],
                            columns=['length', 'upper_case', 'lower_case', 'numbers', 'special_chars', 'log2_prob'])

    dt_prediction = dt_model.predict(features)[0]
    lg_prediction = lg_model.predict(features)[0]

    st.write('**Decision Tree Prediction:**', 'Strong' if dt_prediction else 'Weak')
    st.write('**Logistic Regression Prediction:**', 'Strong' if lg_prediction else 'Weak')
