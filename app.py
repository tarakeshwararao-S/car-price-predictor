import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD SAVED OBJECTS ---------------- #
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title-text {
    font-size: 48px;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
}
.subtitle-text {
    font-size: 20px;
    color: #4a4a4a;
    text-align: center;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.price-box {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    padding: 25px;
    border-radius: 15px;
    color: white;
    font-size: 30px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown('<div class="title-text">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Machine Learning based price estimation</div>', unsafe_allow_html=True)

st.image(
    "https://images.unsplash.com/photo-1503376780353-7e6692767b70",
    use_container_width=True
)

st.markdown("---")

# ---------------- INPUT SECTION ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📋 Enter Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    symboling = st.selectbox("Symboling", [-3, -2, -1, 0, 1, 2, 3])
    fueltype = st.selectbox("Fuel Type", ["gas", "diesel"])
    aspiration = st.selectbox("Aspiration", ["std", "turbo"])
    doornumber = st.selectbox("Door Number", ["two", "four"])
    carbody = st.selectbox("Car Body", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])

with col2:
    drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
    enginelocation = st.selectbox("Engine Location", ["front", "rear"])
    enginetype = st.selectbox("Engine Type", ["ohc", "ohcf", "ohcv", "dohc", "l"])
    cylindernumber = st.selectbox(
        "Cylinders", ["two", "three", "four", "five", "six", "eight", "twelve"]
    )
    fuelsystem = st.selectbox(
        "Fuel System", ["mpfi", "2bbl", "idi", "1bbl", "spdi", "4bbl"]
    )

with col3:
    wheelbase = st.number_input("Wheelbase", value=95.0)
    carlength = st.number_input("Car Length", value=170.0)
    carwidth = st.number_input("Car Width", value=65.0)
    carheight = st.number_input("Car Height", value=54.0)
    curbweight = st.number_input("Curb Weight", value=2000)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ENGINE & PERFORMANCE ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚙️ Engine & Performance")

col4, col5, col6 = st.columns(3)

with col4:
    enginesize = st.number_input("Engine Size", value=120)
    boreratio = st.number_input("Bore Ratio", value=3.0)

with col5:
    stroke = st.number_input("Stroke", value=3.0)
    compressionratio = st.number_input("Compression Ratio", value=9.0)

with col6:
    horsepower = st.number_input("Horsepower", value=100)
    peakrpm = st.number_input("Peak RPM", value=5000)
    citympg = st.number_input("City MPG", value=25)
    highwaympg = st.number_input("Highway MPG", value=30)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DATAFRAME ---------------- #
input_data = {
    'symboling': symboling,
    'fueltype': fueltype,
    'aspiration': aspiration,
    'doornumber': doornumber,
    'carbody': carbody,
    'drivewheel': drivewheel,
    'enginelocation': enginelocation,
    'wheelbase': wheelbase,
    'carlength': carlength,
    'carwidth': carwidth,
    'carheight': carheight,
    'curbweight': curbweight,
    'enginetype': enginetype,
    'cylindernumber': cylindernumber,
    'enginesize': enginesize,
    'fuelsystem': fuelsystem,
    'boreratio': boreratio,
    'stroke': stroke,
    'compressionratio': compressionratio,
    'horsepower': horsepower,
    'peakrpm': peakrpm,
    'citympg': citympg,
    'highwaympg': highwaympg
}

df = pd.DataFrame([input_data])
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.reindex(columns=columns, fill_value=0)
df_scaled = scaler.transform(df_encoded)

# ---------------- PREDICTION ---------------- #
st.markdown("---")
if st.button("🚀 Predict Car Price"):
    prediction = model.predict(df_scaled)[0]
    st.markdown(
        f'<div class="price-box">Estimated Car Price: ₹ {prediction:,.2f}</div>',
        unsafe_allow_html=True
    )

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("© Mini Project | Linear Regression | Streamlit Deployment")
