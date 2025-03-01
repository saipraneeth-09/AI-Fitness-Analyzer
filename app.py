import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding="utf-8")  # Ensure proper encoding
    df = df.dropna()  # Remove missing values
    return df

@st.cache_data
def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical features using Label Encoding
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Encode target variable if it's categorical
    target_encoder = None
    if y.dtype == "object":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, target_encoder

# --- Model Training and Evaluation ---
@st.cache_resource
def train_model(X_train, y_train, model_name):
    if model_name == "SVM":
        model = SVC(random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Invalid model name")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# --- Streamlit App ---
def main():
    st.title("🏋️ Personal Fitness Tracker")

    uploaded_file = st.file_uploader(" Upload your fitness data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write(" **Data Preview:**")
        st.dataframe(df.head())

        target_column = st.selectbox(" Select the target column", df.columns)

        # Preprocessing
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, target_encoder = preprocess_data(df, target_column)

        # Model selection
        model_name = st.selectbox(" Select a model", ["SVM", "Logistic Regression", "Random Forest"])
        model = train_model(X_train_scaled, y_train, model_name)

        # Model evaluation
        accuracy, report = evaluate_model(model, X_test_scaled, y_test)

        st.write(f"###  {model_name} Model Evaluation")
        st.write(f" **Accuracy:** {accuracy:.4f}")
        st.text("📄 **Classification Report:**\n" + report)

        # Prediction interface
        st.subheader("🔮 Make Predictions")
        input_data = {}
        for col in df.drop(columns=[target_column]).columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))
            else:
                input_data[col] = st.selectbox(f"Select {col}", df[col].unique())

        if st.button(" Predict"):
            input_df = pd.DataFrame([input_data])

            # Encode categorical features
            for col, le in label_encoders.items():
                input_df[col] = le.transform(input_df[col])

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            # Decode prediction if target was categorical
            if target_encoder:
                prediction = target_encoder.inverse_transform([prediction])[0]

            st.success(f" **Prediction:** {prediction}")

        # Visualization of Feature Importance (Random Forest only)
        if model_name == "Random Forest":
            st.subheader("Feature Importance (Random Forest)")
            feature_importance = model.feature_importances_
            feature_names = df.drop(columns=[target_column]).columns
            feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
            feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.title("Feature Importance")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(plt)

if __name__ == "__main__":
    main()
