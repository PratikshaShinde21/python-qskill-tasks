import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="QSkill Tasks", layout="wide")

st.title("🎓 QSkill Internship - Slab 1 Tasks")
st.write("Data Analysis | House Price Prediction | Matrix Operations")

# Sidebar menu
option = st.sidebar.selectbox(
    "Choose Task:",
    ["Home", "Task 1: Data Analysis", "Task 2: House Price Prediction", "Task 3: Matrix Tool"]
)

if option == "Home":
    st.header("Welcome to QSkill Tasks! 👋")
    st.write("""
    This application demonstrates:
    - **Task 1:** Data Analysis with Pandas & Matplotlib
    - **Task 2:** House Price Prediction using Linear Regression
    - **Task 3:** Matrix Operations Tool using NumPy
    """)
    
    st.success("✅ All tasks completed successfully!")

elif option == "Task 1: Data Analysis":
    st.header("📊 Task 1: Student Performance Analysis")
    
    # Create sample data
    np.random.seed(42)
    n = 100
    data = {
        "StudentName": [f"Student_{i}" for i in range(1, n + 1)],
        "Math": np.random.randint(40, 100, n),
        "Science": np.random.randint(35, 100, n),
        "English": np.random.randint(30, 100, n),
        "History": np.random.randint(25, 100, n),
        "Computer": np.random.randint(50, 100, n),
    }
    
    df = pd.DataFrame(data)
    df["Total"] = df[["Math", "Science", "English", "History", "Computer"]].sum(axis=1)
    df["Average"] = df["Total"] / 5
    
    st.subheader("📋 Student Data (First 10 rows)")
    st.dataframe(df.head(10))
    
    st.subheader("📈 Statistics")
    st.write(df[["Math", "Science", "English", "History", "Computer"]].describe())
    
    # Charts
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Subject-wise Average**")
        subject_avg = df[["Math", "Science", "English", "History", "Computer"]].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(subject_avg.index, subject_avg.values, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'])
        ax.set_ylabel("Average Marks")
        st.pyplot(fig)
    
    with col2:
        st.write("**Math vs Science**")
        fig, ax = plt.subplots(figsize=(8, 4))
        scatter = ax.scatter(df["Math"], df["Science"], c=df["Average"], cmap="viridis", alpha=0.6)
        ax.set_xlabel("Math Marks")
        ax.set_ylabel("Science Marks")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
    
    st.success(f"✅ Average Score: {df['Average'].mean():.2f}")

elif option == "Task 2: House Price Prediction":
    st.header("🏠 Task 2: House Price Prediction")
    
    # Create sample data
    np.random.seed(0)
    n = 300
    locations = ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"]
    loc_price = {"Mumbai": 15000, "Pune": 9000, "Nagpur": 5000, "Nashik": 4000, "Aurangabad": 3500}
    
    rows = []
    for _ in range(n):
        loc = np.random.choice(locations)
        rooms = np.random.randint(1, 6)
        size_sqft = rooms * np.random.randint(200, 350)
        age = np.random.randint(0, 30)
        parking = np.random.randint(0, 3)
        base = loc_price[loc]
        price = (base * size_sqft / 1000 + rooms * 50000 + parking * 30000 - age * 5000 + np.random.randint(-50000, 50000))
        price = max(price, 200000)
        rows.append([rooms, loc, size_sqft, age, parking, round(price, -3)])
    
    df = pd.DataFrame(rows, columns=["Rooms", "Location", "SizeSqft", "AgeYears", "Parking", "Price"])
    
    st.subheader("📊 House Data")
    st.dataframe(df.head(10))
    
    # Train model
    le = LabelEncoder()
    df["Location_enc"] = le.fit_transform(df["Location"])
    features = ["Rooms", "Location_enc", "SizeSqft", "AgeYears", "Parking"]
    X = df[features]
    y = df["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.subheader("📈 Model Evaluation")
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"₹{mae:,.0f}")
    col2.metric("R² Score", f"{r2:.4f}")
    col3.metric("Accuracy", f"{r2*100:.1f}%")
    
    # Prediction chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_test, y_pred, alpha=0.6)
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2)
    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title("Actual vs Predicted Price")
    st.pyplot(fig)
    
    st.success("✅ Model trained successfully!")

elif option == "Task 3: Matrix Tool":
    st.header("🔢 Task 3: Matrix Operations")
    
    st.subheader("Interactive Matrix Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Matrix A**")
        a_rows = st.number_input("Rows (A):", min_value=1, max_value=5, value=2)
        a_cols = st.number_input("Columns (A):", min_value=1, max_value=5, value=2)
        matrix_a = st.text_area("Enter A (space-separated rows):", "1 2\n3 4")
    
    with col2:
        st.write("**Matrix B**")
        b_rows = st.number_input("Rows (B):", min_value=1, max_value=5, value=2)
        b_cols = st.number_input("Columns (B):", min_value=1, max_value=5, value=2)
        matrix_b = st.text_area("Enter B (space-separated rows):", "5 6\n7 8")
    
    operation = st.selectbox("Choose Operation:", ["Addition", "Subtraction", "Multiplication", "Transpose A", "Determinant A"])
    
    try:
        A = np.array([list(map(float, row.split())) for row in matrix_a.strip().split('\n')])
        B = np.array([list(map(float, row.split())) for row in matrix_b.strip().split('\n')])
        
        if operation == "Addition":
            result = A + B
            st.write("**Result:**")
            st.write(result)
        elif operation == "Subtraction":
            result = A - B
            st.write("**Result:**")
            st.write(result)
        elif operation == "Multiplication":
            result = A @ B
            st.write("**Result:**")
            st.write(result)
        elif operation == "Transpose A":
            result = A.T
            st.write("**Transpose of A:**")
            st.write(result)
        elif operation == "Determinant A":
            if A.shape[0] == A.shape[1]:
                det = np.linalg.det(A)
                st.metric("Determinant |A|", f"{det:.4f}")
            else:
                st.error("Determinant requires a square matrix!")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")