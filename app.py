import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

@st.cache_data
def train_model(data):
    # Load and process data
    data['NumCompaniesWorked'].fillna(data['NumCompaniesWorked'].mean(), inplace=True)
    data['TotalWorkingYears'].fillna(data['TotalWorkingYears'].mean(), inplace=True)
    
    X = data.drop(['Attrition', 'EmployeeID'], axis=1)
    X = pd.get_dummies(X)  # Convert categorical variables to dummy variables
    y = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert target variable to binary format
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns.tolist()


def main():
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://i.hizliresim.com/6s7l04p.jpg');
            background-size: cover;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <style>
        .stApp {
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            margin: 10px 0;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            margin: 10px;
        }
        .result-box {
            background-color: white;
            color: black;
            padding: 20px;
            margin-top: 0px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Employee Attrition Prediction")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            model, feature_columns = train_model(data)
            st.write("Model trained successfully!")
            st.write(f"Features used: {feature_columns}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        
        user_input = {}
        columns = st.columns(3)
        for i, feature in enumerate(feature_columns):
            with columns[i % 3]:
                if feature in ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome', 
                               'NumCompaniesWorked', 'PercentSalaryHike', 'StandardHours', 
                               'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                               'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
                    user_input[feature] = st.number_input(f'{feature}', min_value=0, step=1, value=1)
                elif feature in data.columns:
                    unique_values = data[feature].unique().tolist()
                    user_input[feature] = st.selectbox(f'{feature}', unique_values)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Predict"):
            user_input_df = pd.DataFrame([user_input])
            user_input_df = pd.get_dummies(user_input_df)

            user_input_df = user_input_df.reindex(columns=feature_columns, fill_value=0)
            
            # Make prediction
            prediction = model.predict(user_input_df)
            prediction_prob = model.predict_proba(user_input_df)
            
            if prediction[0] == 1:
                prediction_text = f"The employee is likely to leave.\nProbability of leaving: %{prediction_prob[0][1]*100:.2f}"
            else:
                prediction_text = f"The employee is likely to stay.\nProbability of leaving: %{prediction_prob[0][1]*100:.2f}"
            
            st.session_state["prediction_text"] = prediction_text

    with col2:
        st.markdown(f"<div class='result-box'>{st.session_state.get('prediction_text', '')}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()