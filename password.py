import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Password dataset
passwords = ['123', 'Password1!', 'letmein', 'qwerty123', 'Str0ngPass#', 'hello12345', 'Admin2021!']
tries = [5, 15000, 10, 8000, 200000, 7000, 180000]

data = {
    'password': passwords,
    'length': [len(p) for p in passwords],
    'upper_case': [sum(1 for c in p if c.isupper()) for p in passwords],
    'lower_case': [sum(1 for c in p if c.islower()) for p in passwords],
    'numbers': [sum(1 for c in p if c.isdigit()) for p in passwords],
    'special_chars': [len(re.findall(r'[^a-zA-Z0-9]', p)) for p in passwords],
    'tries': tries
}

pwd_data = pd.DataFrame(data)
pwd_data['log2_prob'] = np.log2(1 / pwd_data['tries'])
SEPARATOR = -10.0
pwd_data['LIKELY'] = (pwd_data['log2_prob'] > SEPARATOR).astype(int)

# Streamlit app
st.title('Password Strength Checker')
st.subheader('Sample Password Dataset:')
st.dataframe(pwd_data)

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log2_prob', y='tries', hue='LIKELY', data=pwd_data)
plt.title('Password Tries vs Log2 Probability')
st.pyplot(plt)

# Prepare data
X = pwd_data[['length', 'upper_case', 'lower_case', 'numbers', 'special_chars', 'log2_prob']]
y = pwd_data['LIKELY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
lg_model = LogisticRegression(max_iter=400)
lg_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
nn_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500)
nn_model.fit(X_train, y_train)

# Accuracy
models = {'Decision Tree': dt_model, 'Logistic Regression': lg_model, 'Random Forest': rf_model, 'Neural Network': nn_model}
for name, model in models.items():
    pred = model.predict(X_test)
    st.write(f"{name} Accuracy: {metrics.accuracy_score(y_test, pred):.2f}")

# Password suggestion system
def generate_strong_password(password):
    if len(password) < 8:
        password += 'Xyz123!@'
    if not re.search(r'[A-Z]', password):
        password += 'A'
    if not re.search(r'[a-z]', password):
        password += 'a'
    if not re.search(r'[0-9]', password):
        password += '1'
    if not re.search(r'[^a-zA-Z0-9]', password):
        password += '!'
    return password

st.subheader('Test Your Password:')
user_password = st.text_input('Enter your password:', type='password')
if user_password:
    strong_password = generate_strong_password(user_password)
    st.write(f'Suggested Strong Password: {strong_password}')

    password_data = pd.DataFrame({
        'length': [len(user_password)],
        'upper_case': [sum(1 for c in user_password if c.isupper())],
        'lower_case': [sum(1 for c in user_password if c.islower())],
        'numbers': [sum(1 for c in user_password if c.isdigit())],
        'special_chars': [len(re.findall(r'[^a-zA-Z0-9]', user_password))],
        'log2_prob': [-12]  # placeholder value
    })
    prediction = rf_model.predict(password_data)
    st.write('Password Strength:', 'Strong' if prediction[0] == 1 else 'Weak')
