import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# ----------------- LOAD & TRAIN MODEL -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Employee_Attrition.csv")
    df.columns = df.columns.str.lower()

    drop_cols = ["employeecount", "standardhours", "over18"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["attrition"] = df["attrition"].astype(int)
    return df

@st.cache_resource
def train_model(df):
    # use only simple numeric features (no encoded text codes)
    feature_cols = [
        "age",
        "monthlyincome",
        "jobsatisfaction",
        "yearsatcompany",
        "worklifebalance",
        "environmentsatisfaction",
        "overtime"          # assume already encoded 0 = No, 1 = Yes
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # scores for dashboard
    X_all_s = scaler.transform(X)
    df_scores = df.copy()
    df_scores["attrition_risk"] = model.predict_proba(X_all_s)[:, 1]
    df_scores["performance_score_percent"]= df_scores["performancerating"] / df_scores["performancerating"].max()

    return model, scaler, feature_cols, df_scores, (acc, prec, rec)

df = load_data()
model, scaler, feature_cols, df_scores, metrics = train_model(df)
acc, prec, rec = metrics

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")
st.sidebar.title("Employee Attrition Analysis")
page = st.sidebar.radio("Navigation", ["Home", "Predict Employee Attrition"])

# ---------- HOME ----------
if page == "Home":
    st.title("Employee Insights Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Precision (attrition=1)", f"{prec:.2%}")
    c3.metric("Recall (attrition=1)", f"{rec:.2%}")

    st.markdown("🔴 High-Risk Employees")
    high_risk = df_scores.sort_values("attrition_risk", ascending=False)
    st.dataframe(
        high_risk[["employeenumber",
                   "monthlyincome",
                   "jobsatisfaction",
                   "yearsatcompany",
                   "attrition_risk",
                   "performance_score_percent"]].head(10)
    )

    st.markdown("🏆 High Performance Score")
    high_perf = df_scores.sort_values("performance_score_percent", ascending=False)
    st.dataframe(
        high_perf[["employeenumber",
                   "monthlyincome",
                   "jobsatisfaction",
                   "yearsatcompany",
                   "attrition_risk",
                   "performance_score_percent"]].head(10)
    )

    st.markdown("😊 High job satisfaction list")
    high_satisfaction = df_scores.sort_values("jobsatisfaction", ascending=False)
    st.dataframe(
        high_satisfaction[["employeenumber",
                           "monthlyincome",
                           "jobsatisfaction",
                           "yearsatcompany",
                           "attrition_risk",
                           "performance_score_percent"]].head(10)
    )

# ---------- PREDICTION PAGE ----------
else:
    st.title("Predict Employee Attrition")
    st.markdown("Enter employee details to get the **attrition risk**.")

    with st.form("predict_form"):
        age = st.number_input(
            "Age", 18, 70, int(df["age"].median())
        )
        monthlyincome = st.number_input(
            "Monthly Income",
            int(df["monthlyincome"].min()),
            int(df["monthlyincome"].max()),
            int(df["monthlyincome"].median())
        )
        jobsatisfaction = st.slider(
            "Job Satisfaction (1–4)", 1, 4, int(df["jobsatisfaction"].median())
        )
        maritalstatus = st.radio(
            "Marital Status", ["Single", "Married", "Divorced"], index=0
        )
        yearsatcompany = st.slider(
            "Years at Company",
            int(df["yearsatcompany"].min()),
            int(df["yearsatcompany"].max()),
            int(df["yearsatcompany"].median())
        )
        worklifebalance = st.slider(
            "Work–Life Balance (1–4)", 1, 4, int(df["worklifebalance"].median())
        )
        environmentsatisfaction = st.slider(
            "Environment Satisfaction (1–4)", 1, 4, int(df["environmentsatisfaction"].median())
        )
        overtime_choice = st.radio(
            "Overtime", ["No", "Yes"], index=0
        )
        # map to same encoding as cleaned data: assume 0 = No, 1 = Yes
        overtime = 1 if overtime_choice == "Yes" else 0

        submit = st.form_submit_button("🔍 Predict Attrition")

    if submit:
        input_data = {
            "age": age,
            "monthlyincome": monthlyincome,
            "jobsatisfaction": jobsatisfaction,
            "maritalstatus": maritalstatus,
            "yearsatcompany": yearsatcompany,
            "worklifebalance": worklifebalance,
            "environmentsatisfaction": environmentsatisfaction,
            "overtime": overtime
        }

        input_df = pd.DataFrame([input_data])[feature_cols]
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("### Prediction Result")

        if pred == 1:
            st.error(
                f"⚠ The employee is **LIKELY TO LEAVE**.\n\n"
                f"Estimated attrition risk: **{prob:.2%}**"
            )
        else:
            st.success(
                f"✔ The employee is **NOT LIKELY TO LEAVE**.\n\n"
                f"Estimated attrition risk: **{prob:.2%}**"
            )
