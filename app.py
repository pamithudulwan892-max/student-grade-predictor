# Streamlit app for the student pass predictor
# run with: streamlit run app.py
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

GITHUB_URL = "https://github.com/pamithudulwan892-max/student-grade-predictor"

st.set_page_config(page_title="Student Pass Predictor", page_icon=":mortar_board:", layout="centered")


@st.cache_resource
def load():
    return joblib.load("model/model.joblib"), json.loads(Path("model/metrics.json").read_text())


model, meta = load()
S = meta["schema"]

LEVEL_5 = {1: "1 - Very low", 2: "2 - Low", 3: "3 - Medium", 4: "4 - High", 5: "5 - Very high"}
STUDY = {1: "Less than 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "More than 10 hours"}
EDU = {0: "None", 1: "Primary (up to 4th grade)", 2: "5th - 9th grade", 3: "Secondary", 4: "Higher education"}
HEALTH = {1: "1 - Very bad", 2: "2 - Bad", 3: "3 - OK", 4: "4 - Good", 5: "5 - Very good"}


def pick(col, label, key, mapping, help=None):
    keys = list(mapping)
    default = int(round(S[key]["default"]))
    return col.selectbox(label, keys, index=keys.index(default), format_func=mapping.get, help=help)


def yes_no(col, label, key):
    return col.radio(label, ["yes", "no"], index=["yes", "no"].index(S[key]["default"]), horizontal=True)


st.title("Student Pass Predictor")
st.caption("Predicts if a student will pass the final exam, using only things known early in the "
           "school year (first period grade, absences, study habits and so on).")

tab_pred, tab_perf, tab_about = st.tabs(["Predict", "Model performance", "About"])

with tab_pred:
    with st.form("inputs"):
        st.subheader("Academics")
        a, b = st.columns(2)
        g1 = a.slider("First period grade (0-20)", 0, 20, int(S["first_period_grade"]["default"]))
        subject = b.selectbox("Subject", S["subject"]["options"])
        study = pick(a, "Weekly study time", "weekly_study_time", STUDY)
        failures = b.number_input("Past class failures", 0, 4, 0)
        absences = a.number_input("Number of absences", 0, 93, int(S["absences"]["default"]))
        higher = yes_no(b, "Wants higher education?", "wants_higher_education")

        st.subheader("Student & family")
        a, b = st.columns(2)
        sex = a.radio("Sex", ["F", "M"], horizontal=True)
        age = b.number_input("Age", 15, 22, int(S["age"]["default"]))
        medu = pick(a, "Mother's education", "mother_education", EDU)
        internet = yes_no(b, "Internet at home?", "internet_at_home")
        schoolsup = yes_no(a, "Extra school support?", "extra_school_support")
        famsup = yes_no(b, "Family educational support?", "family_support")

        st.subheader("Lifestyle")
        a, b, c = st.columns(3)
        goout = pick(a, "Going out with friends", "going_out_with_friends", LEVEL_5)
        walc = pick(b, "Weekend alcohol use", "weekend_alcohol_use", LEVEL_5)
        health = pick(c, "Health status", "health_status", HEALTH)

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        X = pd.DataFrame([{
            "subject": subject, "sex": sex, "age": age, "weekly_study_time": study,
            "past_class_failures": failures, "absences": absences,
            "wants_higher_education": higher, "internet_at_home": internet,
            "extra_school_support": schoolsup, "family_support": famsup,
            "going_out_with_friends": goout, "weekend_alcohol_use": walc,
            "health_status": health, "mother_education": medu, "first_period_grade": g1,
        }])
        proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
        p_pass = proba["Pass"]

        def pct(p):
            return ">99%" if p > 0.99 else "<1%" if p < 0.01 else f"{p:.0%}"

        if p_pass >= 0.5:
            st.success(f"### Likely to pass ({pct(p_pass)} probability)")
        else:
            st.error(f"### At risk of failing ({pct(1 - p_pass)} probability of failing)")
        st.progress(float(p_pass), text=f"Pass probability: {pct(p_pass)}")
        if p_pass < 0.7:
            st.info("In this data the biggest warning signs are a low first period grade and a lot of "
                    "absences.")

with tab_perf:
    st.write(f"**Best model:** {meta['best_model']} (picked using 5-fold cross-validation, {meta['cv_metric']}).")
    st.write("**Test set results** (students the model never saw):")
    for c, (k, v) in zip(st.columns(len(meta["test_metrics"])), meta["test_metrics"].items()):
        c.metric(k.replace("_", " ").upper(), f"{v:.3f}")
    st.write("**Models I compared (cross-validation ROC-AUC):**")
    st.dataframe(pd.Series(meta["cv_scores"], name="CV ROC-AUC").sort_values(ascending=False).round(3))
    for img, cap in [("confusion_matrix.png", "Confusion matrix on the test set"),
                     ("feature_importance.png", "Which inputs matter most")]:
        if Path("assets", img).exists():
            st.image(f"assets/{img}", caption=cap)

with tab_about:
    st.markdown(f"""
**Dataset:** [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
with {meta['n_rows']:,} student records from two Portuguese secondary schools (Math and Portuguese courses).
Source: P. Cortez & A. Silva (2008). License: CC BY 4.0.

**Target:** pass = final grade (G3) of 10 or more out of 20.
I didn't use the second period grade, because it's only known late in the year.

**Code:** [{GITHUB_URL}]({GITHUB_URL})
""")
    st.caption("Made with scikit-learn and Streamlit. This is a learning project, not something to make real decisions about students with.")
