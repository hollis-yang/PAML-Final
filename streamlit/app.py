import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
import sys
import json
from pathlib import Path
import geopandas as gpd

# Add project root to path to allow stage1/stage2 imports

# Add project root to path to allow stage1/stage2 imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage1.predict import TrafficPredictor
from stage2.predict import CrashPredictor, WeatherAblationPredictor

# Enable dark theme for Altair
alt.themes.enable('dark')
# --- Page Config ---
st.set_page_config(
    page_title="NYC Crash Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Premium Gradient Title */
    .gradient-text {
        background: linear-gradient(135deg, #f97316 0%, #facc15 50%, #f97316 100%) !important;
        background-size: 200% auto !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        animation: shine 4s linear infinite;
        display: inline-block !important;
    }
    
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }
    
    .header-container h1 {
        text-align: center !important;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.8rem !important;
        letter-spacing: -0.02em !important;
        text-shadow: 0 10px 30px rgba(249, 115, 22, 0.3) !important;
    }
    
    .header-tagline {
        text-align: center;
    }
    
    .header-tagline span {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        background: rgba(15, 23, 42, 0.6);
        padding: 0.4rem 1.5rem;
        border-radius: 50px;
        border: 1px solid rgba(249, 115, 22, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }

    .header-container {
        padding: 1.5rem 1rem 2rem 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Premium Dark Blue Containers (Targeted) */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.premium-card) {
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
        padding: 24px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .premium-card { display: none; }
    
    /* Smooth button hover */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(45deg, #ea580c, #f97316) !important;
        color: white !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(249, 115, 22, 0.3);
    }
    
    /* Prevent checkbox label wrapping */
    [data-testid="stCheckbox"] label {
        white-space: nowrap !important;
    }
    
    /* Premium Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        color: #94a3b8 !important;
        transition: all 0.3s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #f97316 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f97316 !important;
        border-bottom-color: #f97316 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_models():
    cp = CrashPredictor()
    return TrafficPredictor(), cp, WeatherAblationPredictor(cp)

try:
    traffic_model, crash_model, ablation_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Constants for normalization (calculated from data/data.csv)
TAVG_MEAN, TAVG_STD = 57.07, 16.32
AWND_MEAN, AWND_STD = 5.03, 2.36

# --- Cached Map Data Loading ---
@st.cache_data
def get_raw_data():
    # Use the dataset with date
    df = pd.read_csv("data/data_with_date.csv")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['weekday'] = pd.to_datetime(df['date']).dt.weekday
    
    # Create derived columns needed for filtering and plotting
    if 'PRCP' in df.columns:
        df['is_rain'] = (df['PRCP'] > 0).astype(int)
    if 'SNOW' in df.columns:
        df['is_snow'] = (df['SNOW'] > 0).astype(int)
    if 'traffic_count' in df.columns:
        df['log_traffic_count'] = np.log1p(df['traffic_count'])
        
    return df

@st.cache_data
def get_historical_weather_avg(target_date):
    df = get_raw_data()
    if 'date' not in df.columns or df.empty:
        return 50.0, 0.0, 0.0, 0.0, 5.0
        
    target_doy = target_date.timetuple().tm_yday
    dt_series = pd.to_datetime(df['date'])
    df_doy = dt_series.dt.dayofyear
    
    mask = (abs(df_doy - target_doy) <= 10) | (abs(df_doy - target_doy) >= 355)
    window_df = df[mask]
    
    if window_df.empty:
        return 50.0, 0.0, 0.0, 0.0, 5.0
        
    avg_tavg = window_df['TAVG'].mean() if 'TAVG' in window_df.columns else 50.0
    avg_prcp = window_df['PRCP'].mean() if 'PRCP' in window_df.columns else 0.0
    avg_snow = window_df['SNOW'].mean() if 'SNOW' in window_df.columns else 0.0
    avg_snwd = window_df['SNWD'].mean() if 'SNWD' in window_df.columns else 0.0
    avg_awnd = window_df['AWND'].mean() if 'AWND' in window_df.columns else 5.0
    
    return (
        float(avg_tavg) if not pd.isna(avg_tavg) else 50.0,
        float(avg_prcp) if not pd.isna(avg_prcp) else 0.0,
        float(avg_snow) if not pd.isna(avg_snow) else 0.0,
        float(avg_snwd) if not pd.isna(avg_snwd) else 0.0,
        float(avg_awnd) if not pd.isna(avg_awnd) else 5.0
    )

@st.cache_data
def get_map_shapes():
    shp_path = "data/geo_data/nyc_zip/nyc_zip.shp"
    gdf = gpd.read_file(shp_path)
    if "ZIPCODE" in gdf.columns:
        gdf["zip_code"] = gdf["ZIPCODE"].astype(str)
    elif "zcta" in gdf.columns:
        gdf["zip_code"] = gdf["zcta"].astype(str)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def get_borough_from_zip(z):
    try:
        z_int = int(z)
        if 10000 <= z_int <= 10299: return 'Manhattan'
        if 10300 <= z_int <= 10399: return 'Staten Island'
        if 10400 <= z_int <= 10499: return 'Bronx'
        if 11200 <= z_int <= 11299: return 'Brooklyn'
        if (11000 <= z_int <= 11199) or (11300 <= z_int <= 11499) or (11600 <= z_int <= 11699): return 'Queens'
        return 'Other'
    except:
        return 'Unknown'

def filter_data(df, date_range, time_period, weather, borough, zip_code):
    filtered = df.copy()
    
    # 0. Date Range
    if 'date' in filtered.columns:
        if len(date_range) == 2:
            filtered = filtered[(filtered['date'] >= date_range[0]) & (filtered['date'] <= date_range[1])]
        elif len(date_range) == 1:
            filtered = filtered[filtered['date'] == date_range[0]]
            
    # 1. Time Period (using is_peak)
    if time_period == "Peak":
        filtered = filtered[filtered['is_peak'] == 1]
    elif time_period == "Off-Peak":
        filtered = filtered[filtered['is_peak'] == 0]
        
    # 2. Weather
    if weather == "Rain":
        filtered = filtered[filtered['is_rain'] == 1]
    elif weather == "Snow":
        filtered = filtered[filtered['is_snow'] == 1]
    elif weather == "Fog":
        if 'WT01' in filtered.columns:
            filtered = filtered[filtered['WT01'] == 1]
            
    # 3. Borough filtering
    if borough != "All":
        filtered['borough'] = filtered['zip_code'].apply(get_borough_from_zip)
        filtered = filtered[filtered['borough'] == borough]

    # 4. ZIP Code
    if zip_code:
        filtered = filtered[filtered['zip_code'].astype(str) == str(zip_code)]
        
    return filtered

# --- Header ---
st.markdown("""
<div class="header-container">
    <h1><span class="gradient-text">NYC CRASH PREDICTOR</span></h1>
    <div class="header-tagline"><span>Advanced Analytics & Machine Learning for Urban Safety</span></div>
</div>
""", unsafe_allow_html=True)

# --- Navigation / Tabs ---
# Use Streamlit tabs to simulate navigation between Prediction and Explore Data
tab_pred, tab_explore = st.tabs(["Prediction", "Explore Data"])

with tab_pred:
    # --- Layout: Two Columns ---
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Input Conditions")
        
        # Removed outer container as requested
        st.markdown('<span class="premium-card"></span>', unsafe_allow_html=True)
        
        
        st.markdown("##### Date & Location")
        MIN_DATE = datetime.date.today()
        date_input = st.date_input("Select Date", MIN_DATE, min_value=MIN_DATE)
        
        # Get historical averages for pre-filling
        avg_tavg, avg_prcp, avg_snow, avg_snwd, avg_awnd = get_historical_weather_avg(date_input)
        
        borough_pred = st.selectbox("Borough", options=["All", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], key="pred_borough")
        zip_code = st.text_input("ZIP Code (5-digit, Optional)", value="", placeholder="e.g. 10001", max_chars=5, key="pred_zip")
        time_period = st.radio("Time Period", options=["Peak Hours", "Off-Peak"], horizontal=True)
        
        st.markdown("---")
        st.markdown("##### Weather Conditions")
        c1, c2 = st.columns(2)
        with c1:
            temp = st.number_input("Temperature (°F)", value=round(avg_tavg, 1), step=1.0)
            precip = st.number_input("Precipitation (in)", value=round(avg_prcp, 2), step=0.1)
            snowfall = st.number_input("Snowfall (in)", value=round(avg_snow, 2), step=0.1)
        with c2:
            snow_depth = st.number_input("Snow Depth (in)", value=round(avg_snwd, 2), step=0.1)
            wind_speed = st.number_input("Wind Speed (mph)", value=round(avg_awnd, 1), step=1.0)
        
        st.markdown("---")
        st.markdown("##### Extreme Weather Flags")
        flags_col1, flags_col2 = st.columns(2)
        with flags_col1:
            fog = st.checkbox("Fog")
            heavy_fog = st.checkbox("Heavy Fog")
            thunder = st.checkbox("Thunder")
            ice_pellets = st.checkbox("Ice Pellets")
        with flags_col2:
            hail = st.checkbox("Hail")
            glaze = st.checkbox("Glaze")
            blowing_snow = st.checkbox("Blowing Snow")
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("Predict", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")
        with st.container(border=True):
            st.markdown('<span class="premium-card"></span>', unsafe_allow_html=True)
            
            if predict_button:
                valid_pred = True
                if zip_code and borough_pred != "All":
                    zip_borough = get_borough_from_zip(zip_code)
                    if zip_borough != 'Unknown' and zip_borough != borough_pred:
                        st.error(f"ZIP Code {zip_code} belongs to {zip_borough}, not {borough_pred}. Please correct your input.")
                        valid_pred = False

                if valid_pred:
                    # 1. Prepare inputs for Stage 1
                    # Standardize weekday to 0-6 (Streamlit's date.weekday() is 0=Mon)
                    wd = date_input.weekday()
                    peak_val = 1 if time_period == "Peak Hours" else 0
                    
                    # Z-score normalization for TAVG and AWND
                    tavg_z = (temp - TAVG_MEAN) / TAVG_STD
                    awnd_z = (wind_speed - AWND_MEAN) / AWND_STD
                    
                    # Log transforms for precip, snow, etc.
                    log_prcp = np.log1p(precip)
                    log_snow = np.log1p(snowfall)
                    log_snwd = np.log1p(snow_depth)
                    
                    # Binary weather flags
                    is_rain = 1 if precip > 0 else 0
                    is_snow = 1 if snowfall > 0 else 0
                    has_snwd = 1 if snow_depth > 0 else 0
                    
                    # NOAA WT flags mapping
                    wt_record = {
                        "WT01": 1 if fog else 0,
                        "WT02": 1 if heavy_fog else 0,
                        "WT03": 1 if thunder else 0,
                        "WT04": 1 if ice_pellets else 0,
                        "WT06": 1 if glaze else 0,
                        "WT08": 1 if blowing_snow else 0
                    }
                    
                    input_record = {
                        "zip_code": str(zip_code) if zip_code else "10001",
                        "weekday": wd,
                        "is_peak": peak_val,
                        "tavg_z": tavg_z,
                        "awnd_z": awnd_z,
                        "is_rain": is_rain,
                        "log_prcp": log_prcp,
                        "is_snow": is_snow,
                        "log_snow": log_snow,
                        "has_snow_depth": has_snwd,
                        "log_snwd": log_snwd,
                        **wt_record
                    }

                    # 2. Stage 1 Inference: Predict Traffic Volume
                    with st.spinner("Estimating traffic volume..."):
                        t1_preds = traffic_model.predict_one(input_record)
                        estimated_volume = t1_preds["traffic_ridge"]
                        # Stage 2 expects log_traffic_count
                        input_record["log_traffic_count"] = np.log1p(estimated_volume)

                    # 3. Stage 2 Inference: Predict Crash Count
                    with st.spinner("Predicting crash frequency..."):
                        c2_preds = crash_model.predict_one(input_record)
                        predicted_crashes = c2_preds["mu_nb"]
                        lo95, hi95 = c2_preds["nb_ci95"]
                    
                    # Determine color based on risk level
                    if predicted_crashes < 0.5:
                        res_color = "#22c55e" # Green (Low Risk)
                    elif predicted_crashes < 1.5:
                        res_color = "#f97316" # Orange (Medium Risk)
                    else:
                        res_color = "#ef4444" # Red (High Risk)
                    
                    # Prominent Predicted Crash Count
                    st.markdown(f"<h1 style='text-align: center; color: {res_color}; margin-bottom: 0px;'>Predicted Crash Count: {predicted_crashes:.2f}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-top: 0px;'>95% Confidence Interval: [{lo95}, {hi95}]</p>", unsafe_allow_html=True)
                    
                    # Smaller Estimated Traffic Volume
                    st.markdown(f"<h4 style='text-align: center; color: gray; margin-top: 10px;'>Estimated Traffic Volume: {int(estimated_volume):,} vehicles</h4>", unsafe_allow_html=True)
                    
                    # Model explanation
                    st.info("This prediction is generated using a two-stage model. Traffic volume is first estimated based on time, location, and weather, and then used to predict crash counts.\n\n*(Note: 'Traffic volume' is an area-weighted proxy derived from taxi data to indicate relative activity, rather than an actual vehicle count.)*")
                    
                    # Input summary box
                    st.markdown("---")
                    st.markdown("##### Input Summary")
                    st.write(f"**ZIP Code:** {zip_code if zip_code else '10001'}  |  **Date:** {date_input} (Weekday: {wd})  |  **Time:** {time_period}")
                    st.write(f"**Weather:** {temp}°F, {precip}in precip, {wind_speed}mph wind")
                    
                    flags_selected = [flag_name for flag_val, flag_name in zip(
                        [fog, heavy_fog, thunder, ice_pellets, hail, glaze, blowing_snow],
                        ["Fog", "Heavy Fog", "Thunder", "Ice Pellets", "Hail", "Glaze", "Blowing Snow"]
                    ) if flag_val]
                    
                    if flags_selected:
                        st.write(f"**Active Weather Flags:** {', '.join(flags_selected)}")
                    else:
                        st.write("**Active Weather Flags:** None")
                    
                    st.markdown("---")
                    st.markdown("##### Comparison: Poisson vs Negative Binomial")
                    comp_df = pd.DataFrame({
                        "Model": ["Poisson", "Negative Binomial"],
                        "Predicted Count": [c2_preds["mu_poisson"], c2_preds["mu_nb"]]
                    })
                    
                    # Use Altair to give each model a distinct color
                    compare_chart = alt.Chart(comp_df).mark_bar().encode(
                        x=alt.X("Model:N", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Predicted Count:Q"),
                        color=alt.Color("Model:N", scale=alt.Scale(
                            domain=["Poisson", "Negative Binomial"],
                            range=["#1f77b4", "#f97316"] # Classic Blue for Poisson, Orange for NB
                        ), legend=None)
                    ).properties(height=300)
                    
                    st.altair_chart(compare_chart, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("##### Extreme Weather Impact (Ablation Analysis)")
                    
                    with st.spinner("Analyzing weather impacts..."):
                        ablation_results = ablation_model.predict(input_record)
                        ablation_df = pd.DataFrame(ablation_results)
                        
                        chart_df = ablation_df.copy()
                        chart_df["scenario"] = chart_df["scenario"].str.replace(r"\s+\(WT\d+\)$", "", regex=True)
                        chart_df["scenario"] = chart_df["scenario"].replace("No Flags", "No extreme weather")
                        chart_df["delta_text"] = chart_df["delta_pct"].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
                        chart_df["color"] = chart_df["delta_pct"].apply(
                            lambda x: "#ef4444" if x > 0 else ("#22c55e" if x < 0 else "#94a3b8")
                        )
                        
                        base = alt.Chart(chart_df).encode(
                            x=alt.X("delta_pct:Q", title="Impact on Crash Count (%)"),
                            y=alt.Y("scenario:N", sort="-x", title=""),
                            tooltip=["scenario", "mu_nb", "delta", "delta_pct"]
                        )
                        
                        bars = base.mark_bar().encode(
                            color=alt.Color("color:N", scale=None)
                        )
                        
                        text = base.mark_text(
                            align='left',
                            baseline='middle',
                            dx=3,
                            color='white'
                        ).encode(text='delta_text:N')
                        
                        st.altair_chart((bars + text).properties(height=300), use_container_width=True)
                        st.caption("Shows the percentage change in expected crashes if each extreme weather condition were present, holding all other inputs constant.")
                    
            else:
                # Show a placeholder instruction when the button hasn't been clicked yet
                st.info("Please fill in the input conditions on the left and click 'Predict' to see the results.")

with tab_explore:
    # --- Layout: Two Columns ---
    exp_col1, exp_col2 = st.columns([1, 2.5], gap="large")

    with exp_col1:
        st.subheader("Filters")
        
        with st.container(border=True):
            # 1. Date Range
            MIN_DATE = datetime.date(2020, 1, 1)
            MAX_DATE = datetime.date(2025, 12, 31)
            end_date = min(datetime.date.today(), MAX_DATE)
            start_date = max(end_date - datetime.timedelta(days=30), MIN_DATE)
            date_range = st.date_input("Date Range", (start_date, end_date))
            
            if isinstance(date_range, tuple) and len(date_range) > 0:
                if date_range[0] < MIN_DATE or (len(date_range) == 2 and date_range[1] > MAX_DATE):
                    st.error(f"Please select dates between {MIN_DATE} and {MAX_DATE}.")
            
            # 2. Time Period
            time_period_filter = st.selectbox("Time Period", options=["All", "Peak", "Off-Peak"])
            
            # 3. Weather
            weather_filter = st.selectbox("Weather", options=["All", "Rain", "Snow", "Fog", "Clear"])
            
            # 4. Borough
            borough_filter = st.selectbox("Borough", options=["All", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
            
            # 5. Zipcode
            zip_filter = st.text_input("Zipcode (Optional)", value="", placeholder="e.g. 10001", max_chars=5)
            
            st.markdown("<br>", unsafe_allow_html=True)
            apply_filters = st.button("Apply Filters", type="primary", use_container_width=True)

    with exp_col2:
        st.subheader("Visual Analysis")
        
        # Load data once
        try:
            full_df = get_raw_data()
            if apply_filters:
                valid = True
                
                if isinstance(date_range, tuple) and len(date_range) > 0:
                    if date_range[0] < MIN_DATE or (len(date_range) == 2 and date_range[1] > MAX_DATE):
                        st.error(f"Cannot apply filters: Date range must be between {MIN_DATE} and {MAX_DATE}.")
                        valid = False

                if zip_filter and borough_filter != "All":
                    zip_borough = get_borough_from_zip(zip_filter)
                    if zip_borough != 'Unknown' and zip_borough != borough_filter:
                        st.error(f"ZIP Code {zip_filter} belongs to {zip_borough}, not {borough_filter}. Please correct your input.")
                        valid = False

                if valid:
                    df_to_plot = filter_data(full_df, date_range, time_period_filter, weather_filter, borough_filter, zip_filter)
                    st.toast(f"Filters applied! Showing {len(df_to_plot):,} records.")
                else:
                    df_to_plot = pd.DataFrame()
            else:
                df_to_plot = full_df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            df_to_plot = pd.DataFrame()

        # 1. Crash Density by ZIP Code
        with st.container(border=True):
            st.markdown("#### Average Crash Density")
            with st.spinner("Updating map..."):
                try:
                    map_gdf = get_map_shapes()
                    # Aggregate filtered data
                    agg_filtered = df_to_plot.groupby("zip_code")["crash_count"].mean().reset_index()
                    agg_filtered.columns = ["zip_code", "avg_crashes"]
                    agg_filtered["zip_code"] = agg_filtered["zip_code"].astype(str)
                    
                    # Merge with shapes
                    map_merged = map_gdf.merge(agg_filtered, on="zip_code", how="left").fillna(0)
                    
                    # Calculate dynamic center and zoom
                    map_center = {"lat": 40.7128, "lon": -74.0060}
                    map_zoom = 9
                    
                    if zip_filter:
                        selected_zip_gdf = map_gdf[map_gdf["zip_code"] == str(zip_filter)]
                        if not selected_zip_gdf.empty:
                            centroid = selected_zip_gdf.geometry.centroid.iloc[0]
                            map_center = {"lat": centroid.y, "lon": centroid.x}
                            map_zoom = 12
                    
                    import plotly.express as px
                    fig = px.choropleth_mapbox(
                        map_merged,
                        geojson=json.loads(map_merged.to_json()),
                        locations="zip_code",
                        featureidkey="properties.zip_code",
                        color="avg_crashes",
                        color_continuous_scale="YlOrRd",
                        range_color=(0, map_merged["avg_crashes"].quantile(0.98) if not map_merged["avg_crashes"].empty else 1),
                        mapbox_style="carto-positron",
                        zoom=map_zoom,
                        center=map_center,
                        opacity=0.7,
                        hover_name="zip_code",
                        hover_data={"avg_crashes": ":.3f", "zip_code": False},
                        labels={"avg_crashes": "Avg Crashes"},
                        height=500
                    )
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load map: {e}")
            st.caption("Choropleth map showing engineered crash density averages.")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # 2. Crash Heatmap by Hour & Day
        with st.container(border=True):
            st.markdown("#### Crash Heatmap by Hour & Day")
            
            if not df_to_plot.empty:
                days_map = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
                df_to_plot['Day'] = df_to_plot['weekday'].map(days_map)
                
                # to representative hours.
                peak_data = df_to_plot[df_to_plot['is_peak'] == 1].groupby('Day')['crash_count'].mean().reset_index()
                off_data = df_to_plot[df_to_plot['is_peak'] == 0].groupby('Day')['crash_count'].mean().reset_index()
                
                heatmap_rows = []
                for d in days_map.values():
                    p_val = peak_data[peak_data['Day'] == d]['crash_count'].values[0] if d in peak_data['Day'].values else 0
                    o_val = off_data[off_data['Day'] == d]['crash_count'].values[0] if d in off_data['Day'].values else 0
                    
                    for h in range(24):
                        # Map to representative hours based on the project's Peak definition (7-9, 16-19)
                        if (7 <= h <= 9) or (16 <= h <= 19):
                            heatmap_rows.append({"Day": d, "Hour": h, "Crashes": p_val})
                        else:
                            heatmap_rows.append({"Day": d, "Hour": h, "Crashes": o_val})
                
                df_heat = pd.DataFrame(heatmap_rows)
                
                heat = alt.Chart(df_heat).mark_rect().encode(
                    x=alt.X('Day:O', sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
                    y=alt.Y('Hour:O', title="Hour of Day"),
                    color=alt.Color('Crashes:Q', scale=alt.Scale(scheme='orangered'), title="Avg Crashes")
                ).properties(height=350)
                st.altair_chart(heat, use_container_width=True)
            else:
                st.warning("No data available for heatmap.")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # 3. Top 10 ZIP Codes by Crash Count
        with st.container(border=True):
            st.markdown("#### Top 10 ZIP Codes by Crash Count")
            if not df_to_plot.empty:
                zip_dist = df_to_plot.groupby("zip_code")["crash_count"].sum().reset_index()
                zip_dist = zip_dist.sort_values("crash_count", ascending=False).head(10)
                zip_dist["zip_code"] = zip_dist["zip_code"].astype(str)
                
                bar = alt.Chart(zip_dist).mark_bar(color='#1f77b4').encode(
                    x=alt.X('zip_code:N', sort='-y', axis=alt.Axis(labelAngle=-45), title="ZIP Code"),
                    y=alt.Y('crash_count:Q', title="Total Crashes")
                ).properties(height=350)
                st.altair_chart(bar, use_container_width=True)
            else:
                st.warning("No data available for ZIP distribution.")
            st.caption("Displays the distribution of crash counts across ZIP codes based on active filters.")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # 4. Traffic Volume vs. Crash Count
        with st.container(border=True):
            st.markdown("#### Traffic Volume vs Crash Count")
            if not df_to_plot.empty:
                # We use a sample for the scatter plot to maintain performance if data is huge
                sample_df = df_to_plot.sample(min(2000, len(df_to_plot)))
                # Convert back to real counts
                sample_df['Traffic Volume'] = np.expm1(sample_df['log_traffic_count'])
                sample_df['Crashes'] = sample_df['crash_count']
                
                scatter = alt.Chart(sample_df).mark_circle(size=60, opacity=0.4).encode(
                    x=alt.X('Traffic Volume:Q', title='Estimated Traffic Volume'),
                    y=alt.Y('Crashes:Q', title='Crash Count'),
                    color=alt.Color('is_peak:N', scale=alt.Scale(domain=[0, 1], range=['#38bdf8', '#ff4b4b']), legend=alt.Legend(title="Peak Status", values=[0, 1], labelExpr="datum.value == 1 ? 'Peak' : 'Off-Peak'")),
                    tooltip=['zip_code', 'Traffic Volume', 'Crashes']
                ).properties(height=400).interactive()
                
                st.altair_chart(scatter, use_container_width=True)
                st.caption("Correlation between estimated traffic volume and crash frequency. Blue: Off-Peak, Red: Peak.\n\n*(Note: 'Traffic volume' is an area-weighted proxy derived from taxi data to indicate relative activity, rather than an actual vehicle count.)*")
            else:
                st.warning("No data available for correlation chart.")
