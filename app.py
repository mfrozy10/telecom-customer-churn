import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Load model & reference columns ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_ref_cols():
    df = pd.read_csv("dataset/telecom_churn_processed.csv")
    return [c for c in df.columns if c != "Churn"]

model = load_model()
REF_COLS = load_ref_cols()

# ── Helper: convert raw inputs → 46-col feature vector ─────────────────────────
def build_features(inp: dict) -> pd.DataFrame:
    row = {}

    # ── Numerical ──────────────────────────────────────────────────────────────
    row["Age"]                                = inp["age"]
    row["Number of Dependents"]               = inp["dependents"]
    row["Number of Referrals"]                = inp["referrals"]
    row["Tenure in Months"]                   = inp["tenure"]
    row["Avg Monthly Long Distance Charges"]  = inp["ld_charges"] if inp["phone_service"] == "Yes" else 0.0
    row["Avg Monthly GB Download"]            = inp["gb_download"] if inp["internet_service"] == "Yes" else 0.0
    row["Monthly Charge"]                     = inp["monthly_charge"]
    row["Total Charges"]                      = inp["total_charges"]
    row["Total Refunds"]                      = inp["total_refunds"]
    row["Total Extra Data Charges"]           = inp["extra_data_charges"]
    row["Total Long Distance Charges"]        = inp["total_ld_charges"]
    row["Total Revenue"]                      = inp["total_revenue"]

    # ── Label-encoded binary (mapping verified against training data) ───────────
    row["Gender"]           = 0 if inp["gender"] == "Female" else 1
    row["Married"]          = 0 if inp["married"] == "Yes"   else 1
    row["Phone Service"]    = 0 if inp["phone_service"] == "Yes" else 1
    row["Multiple Lines"]   = 0 if (inp["phone_service"] == "No" or inp["multiple_lines"] == "No") else 1
    row["Internet Service"] = 0 if inp["internet_service"] == "Yes" else 1
    row["Paperless Billing"]= 0 if inp["paperless"] == "Yes" else 1

    # ── OHE: Offer (base = No Offer → all zeros) ───────────────────────────────
    for letter in ["A", "B", "C", "D", "E"]:
        row[f"Offer_Offer {letter}"] = 1 if inp["offer"] == f"Offer {letter}" else 0

    # ── OHE: Internet Type (base = No Internet → all zeros) ───────────────────
    for itype in ["Cable", "DSL", "Fiber Optic"]:
        col = f"Internet Type_{itype}"
        row[col] = 1 if (inp["internet_service"] == "Yes" and inp["internet_type"] == itype) else 0

    # ── OHE: Boolean service pairs (_No / _Yes) ────────────────────────────────
    svc_cols = [
        ("online_security",        "Online Security"),
        ("online_backup",          "Online Backup"),
        ("device_protection",      "Device Protection Plan"),
        ("tech_support",           "Premium Tech Support"),
        ("streaming_tv",           "Streaming TV"),
        ("streaming_movies",       "Streaming Movies"),
        ("streaming_music",        "Streaming Music"),
        ("unlimited_data",         "Unlimited Data"),
    ]
    for key, label in svc_cols:
        if inp["internet_service"] == "No":
            val = "No"
        else:
            val = inp[key]
        row[f"{label}_No"]  = 1 if val == "No"  else 0
        row[f"{label}_Yes"] = 1 if val == "Yes" else 0

    # ── OHE: Contract (base = Month-to-Month → all zeros) ─────────────────────
    row["Contract_One Year"] = 1 if inp["contract"] == "One Year"  else 0
    row["Contract_Two Year"] = 1 if inp["contract"] == "Two Year"  else 0

    # ── OHE: Payment Method (base = Bank Withdrawal → all zeros) ──────────────
    row["Payment Method_Credit Card"]   = 1 if inp["payment"] == "Credit Card"   else 0
    row["Payment Method_Mailed Check"]  = 1 if inp["payment"] == "Mailed Check"  else 0

    return pd.DataFrame([row])[REF_COLS]


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📡 Customer Churn Predictor")
st.caption("Telecom Customer · XGBoost Model · ROC-AUC 0.9027")
st.divider()

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    # ── Column 1: Personal & Account ──────────────────────────────────────────
    with col1:
        st.subheader("Personal Info")
        gender   = st.selectbox("Gender", ["Female", "Male"])
        age      = st.number_input("Age", min_value=19, max_value=80, value=37)
        married  = st.radio("Married", ["Yes", "No"], horizontal=True)
        deps     = st.number_input("Number of Dependents", min_value=0, max_value=9, value=0)
        refs     = st.number_input("Number of Referrals", min_value=0, max_value=11, value=0)

        st.subheader("Account & Contract")
        tenure   = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
        offer    = st.selectbox("Last Offer", ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
        paperless= st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
        payment  = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])

    # ── Column 2: Services ────────────────────────────────────────────────────
    with col2:
        st.subheader("Phone Service")
        phone_svc = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
        multi_ln  = st.radio("Multiple Lines", ["Yes", "No"], horizontal=True,
                             disabled=(phone_svc == "No"))
        ld_charges= st.number_input("Avg Monthly Long Distance Charges ($)",
                                    min_value=0.0, max_value=50.0, value=25.0, step=0.5,
                                    disabled=(phone_svc == "No"))

        st.subheader("Internet Service")
        internet_svc  = st.radio("Internet Service", ["Yes", "No"], horizontal=True)
        internet_type = st.selectbox("Internet Type", ["Cable", "DSL", "Fiber Optic"],
                                     disabled=(internet_svc == "No"))
        gb_download   = st.number_input("Avg Monthly GB Download",
                                        min_value=0.0, max_value=85.0, value=15.0, step=1.0,
                                        disabled=(internet_svc == "No"))

        st.subheader("Add-on Services")
        svc_disabled = (internet_svc == "No")
        online_sec  = st.radio("Online Security",       ["Yes", "No"], horizontal=True, disabled=svc_disabled)
        online_bkp  = st.radio("Online Backup",         ["Yes", "No"], horizontal=True, disabled=svc_disabled)
        device_prot = st.radio("Device Protection Plan",["Yes", "No"], horizontal=True, disabled=svc_disabled)
        tech_sup    = st.radio("Premium Tech Support",  ["Yes", "No"], horizontal=True, disabled=svc_disabled)
        stream_tv   = st.radio("Streaming TV",          ["Yes", "No"], horizontal=True, disabled=svc_disabled)
        stream_mov  = st.radio("Streaming Movies",      ["Yes", "No"], horizontal=True, disabled=svc_disabled)
        stream_mus  = st.radio("Streaming Music",       ["Yes", "No"], horizontal=True, disabled=svc_disabled)
        unltd_data  = st.radio("Unlimited Data",        ["Yes", "No"], horizontal=True, disabled=svc_disabled)

    # ── Column 3: Financials ──────────────────────────────────────────────────
    with col3:
        st.subheader("Financials")
        monthly_charge    = st.number_input("Monthly Charge ($)",             min_value=-10.0, max_value=120.0, value=65.0, step=0.5)
        total_charges     = st.number_input("Total Charges ($)",              min_value=0.0,   max_value=9000.0,value=800.0, step=10.0)
        total_refunds     = st.number_input("Total Refunds ($)",              min_value=0.0,   max_value=50.0,  value=0.0,  step=0.5)
        extra_data        = st.number_input("Total Extra Data Charges ($)",   min_value=0,     max_value=200,   value=0)
        total_ld          = st.number_input("Total Long Distance Charges ($)",min_value=0.0,   max_value=4000.0,value=200.0, step=10.0)
        total_rev         = st.number_input("Total Revenue ($)",              min_value=0.0,   max_value=12000.0,value=1000.0,step=10.0)

        st.divider()
        submitted = st.form_submit_button("🎯  Predict Churn", use_container_width=True, type="primary")

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    inputs = {
        "gender":           gender,
        "age":              age,
        "married":          married,
        "dependents":       deps,
        "referrals":        refs,
        "tenure":           tenure,
        "contract":         contract,
        "offer":            offer,
        "paperless":        paperless,
        "payment":          payment,
        "phone_service":    phone_svc,
        "multiple_lines":   multi_ln,
        "ld_charges":       ld_charges,
        "internet_service": internet_svc,
        "internet_type":    internet_type,
        "gb_download":      gb_download,
        "online_security":  online_sec,
        "online_backup":    online_bkp,
        "device_protection":device_prot,
        "tech_support":     tech_sup,
        "streaming_tv":     stream_tv,
        "streaming_movies": stream_mov,
        "streaming_music":  stream_mus,
        "unlimited_data":   unltd_data,
        "monthly_charge":   monthly_charge,
        "total_charges":    total_charges,
        "total_refunds":    total_refunds,
        "extra_data_charges": extra_data,
        "total_ld_charges": total_ld,
        "total_revenue":    total_rev,
    }

    features = build_features(inputs)
    pred     = model.predict(features)[0]
    prob     = float(model.predict_proba(features)[0][1])

    st.divider()
    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if pred == 1:
            st.error(f"### ⚠️ CHURN RISK DETECTED")
            st.markdown(f"**Churn probability: {prob*100:.1f}%**")
            st.markdown("This customer is likely to leave. Consider a retention offer.")
        else:
            st.success(f"### ✅ NOT CHURN")
            st.markdown(f"**Churn probability: {prob*100:.1f}%**")
            st.markdown("This customer is likely to stay. Keep up the good service!")

    with gauge_col:
        st.markdown("**Churn Probability Gauge**")
        color = "🔴" if prob >= 0.5 else ("🟡" if prob >= 0.3 else "🟢")
        st.progress(prob, text=f"{color}  {prob*100:.1f}%")
        st.caption("🟢 Low risk (<30%)   🟡 Medium (30–50%)   🔴 High (≥50%)")

    # ── Key driver insight ─────────────────────────────────────────────────────
    with st.expander("🔍 Top factors influencing this model (global feature importance)"):
        top_features = {
            "Contract: Two Year":          "21.8% — Longest contracts are the strongest churn reducer",
            "Premium Tech Support: No":    "15.6% — Lack of tech support significantly raises churn risk",
            "Online Security: No":         "13.4% — No security service correlates with higher churn",
            "Contract: One Year":          "11.9% — Annual contracts also reduce churn vs month-to-month",
            "Internet Type: Fiber Optic":  "10.3% — Fiber Optic customers churn more despite premium service",
            "Number of Referrals":          "6.7% — More referrals → higher loyalty",
            "Payment: Credit Card":         "5.5% — Credit card payers show moderate churn differences",
            "Premium Tech Support: Yes":    "5.5% — Having tech support is protective",
            "Streaming Movies: Yes":        "4.8% — Value-added streaming reduces churn tendency",
            "Number of Dependents":         "4.6% — Family dependents correlate with loyalty",
        }
        for feat, desc in top_features.items():
            st.markdown(f"- **{feat}** — {desc}")

    # ── Debug: raw feature vector (collapsed) ─────────────────────────────────
    with st.expander("🧪 Raw feature vector sent to model (for debugging)"):
        st.dataframe(features.T.rename(columns={0: "value"}), use_container_width=True)
