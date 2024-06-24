from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load the dataset from CSV
data = pd.read_csv('insurance.csv')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Health_Status', 'Financial_Goals', 'Marital_Status', 'Risk_Tolerance', 'Future_Financial_Needs', 'Life_Insurance_Type']

for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Split features and target variable
X = data.drop(columns=['Eligibility', 'Life_Insurance_Type'])
y_eligibility = data['Eligibility']
y_insurance_type = data['Life_Insurance_Type']

# Train decision tree model for eligibility
model_eligibility = DecisionTreeClassifier()
model_eligibility.fit(X, y_eligibility)

# Train decision tree model for insurance type
model_insurance_type = DecisionTreeClassifier()
model_insurance_type.fit(X, y_insurance_type)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from frontend
    input_data = request.json
    
    # Preprocess input data
    input_df = pd.DataFrame(input_data, index=[0])
    for col in categorical_cols[:-1]:  # Exclude 'Life_Insurance_Type'
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Make prediction for eligibility
    predicted_eligibility = model_eligibility.predict(input_df)[0]
    if predicted_eligibility == 1:
            eligibility = "Eligible"
    elif predicted_eligibility == 0:
            eligibility = "Not Eligible"
    else:
            eligibility = "Can't predict"


    
    # Make prediction for insurance type
    predicted_insurance_type = model_insurance_type.predict(input_df)[0]
    predicted_insurance_type = label_encoders['Life_Insurance_Type'].inverse_transform([predicted_insurance_type])[0]

    return jsonify({'eligibility': eligibility, 'predicted_insurance_type': predicted_insurance_type})

if __name__ == '__main__':
    app.run(debug=True)
