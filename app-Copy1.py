import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Config ---
st.set_page_config(page_title="🌿 Biochar Predictor", layout="wide")

# --- Header ---
st.title("🌿 Biochar Impact Prediction Tool")
st.markdown(
    """
    <div style='font-size: 18px; margin-bottom: 20px; color: #4B8F8C;'>
        Welcome! This tool helps you estimate <b>Crop Yield</b>, <b>GWP</b> (Global Warming Potential),
        and <b>GHGI</b> (Greenhouse Gas Intensity) based on your climate, region, soil, crop and biochar input features. You can fill in the known values and then click on the Predict button to see the results.
    </div>
    """,
    unsafe_allow_html=True
)

# --- Footer Credit ---
st.markdown(
    "<hr><div style='text-align: right; font-size: 14px; color: gray;'>Made with ❤️ by Mansi Gupta, IIT Guwahati</div>",
    unsafe_allow_html=True
)

# --- Load Model ---
model_c = joblib.load("best_model_crop_gwp.pkl")  # Predicts CropYield_T and GWP_T

# --- Load Dataset for Feature Info ---
df = pd.read_csv('cleaned_features.csv', encoding='latin1')

# --- Drop Target Columns if Present ---
target_cols = ['CropYield_T', 'GWP_T', 'GHGI_T']
df_features = df.loc[:, ~df.columns.isin(target_cols)]

# --- Separate Feature Types ---
numerical_cols = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()

# --- User Input Form ---
st.subheader("🔧 Enter Feature Values Below")
user_input = {}

# --- Numeric Inputs ---
st.markdown("### 🔢 Numerical Inputs")
for i in range(0, len(numerical_cols), 3):
    cols = st.columns(3)
    for j, col in enumerate(numerical_cols[i:i+3]):
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        min_val = float(col_data.min())
        max_val = float(col_data.max())
        label = f"{col} ({min_val:.2f} - {max_val:.2f})"
        user_input[col] = cols[j].number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=None,
            format="%.4f"
        )

# --- Categorical Inputs ---
st.markdown("### 🧬 Categorical Inputs")
for col in categorical_cols:
    options = df[col].dropna().unique().tolist()
    user_input[col] = st.selectbox(f"{col} (optional)", [""] + options)
    if user_input[col] == "":
        user_input[col] = np.nan

# --- Prepare DataFrame ---
input_df = pd.DataFrame([user_input])
input_df = input_df.drop(columns=target_cols, errors='ignore')

# --- Predict Button ---
if st.button("🔮 Predict"):
    try:
        # Ensure input_df only has model features
        input_df = input_df[df_features.columns]

        prediction = model_c.predict(input_df)[0]
        crop_yield = prediction[0]
        gwp = prediction[1]
        ghgi = gwp / crop_yield if crop_yield != 0 else np.nan

        st.markdown("## 📈 Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.success(f"🌾 Crop Yield (T): {crop_yield:.2f}")
        col2.info(f"🌍 GWP_T: {gwp:.2f}")
        col3.warning(f"🌎 GHGI_T: {ghgi:.4f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
