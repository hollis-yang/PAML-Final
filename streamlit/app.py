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
from stage2.predict import CrashPredictor

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
        background: linear-gradient(90deg, #f97316 0%, #fb923c 50%, #f97316 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: #f97316 !important;
        display: inline-block !important;
    }
    
    .header-container h1 {
        text-align: center !important;
        margin-bottom: 0.2rem !important;
        letter-spacing: -0.02em !important;
        color: white !important; /* Base color */
    }
    
    .header-tagline {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 400;
        margin-bottom: 0;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .header-container {
        padding: 0.5rem 1rem;
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
    return TrafficPredictor(), CrashPredictor()

try:
    traffic_model, crash_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Constants for normalization (calculated from data/data.csv)
TAVG_MEAN, TAVG_STD = 57.07, 16.32
AWND_MEAN, AWND_STD = 5.03, 2.36

# --- Cached Map Data Loading ---
@st.cache_data
def get_raw_data():
    # Use the engineered dataset as requested
    df = pd.read_csv("data/data_engineering.csv")
    # Convert log counts back to real counts for visualization
    if 'log_crash_count' in df.columns:
        df['crash_count'] = np.expm1(df['log_crash_count'])
    return df

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

def filter_data(df, time_period, weather, zip_code):
    filtered = df.copy()
    
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
            
    # 3. ZIP Code
    if zip_code:
        filtered = filtered[filtered['zip_code'].astype(str) == str(zip_code)]
        
    return filtered

# --- Header ---
st.markdown("""
<div class="header-container">
    <h1><span class="gradient-text">NYC CRASH PREDICTOR</span></h1>
    <div class="header-tagline">Advanced Analytics & Machine Learning for Urban Safety</div>
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
        date_input = st.date_input("Select Date", datetime.date.today())
        zip_code = st.text_input("ZIP Code (5-digit)", value="", placeholder="e.g. 10001", max_chars=5)
        time_period = st.radio("Time Period", options=["Peak Hours", "Off-Peak"], horizontal=True)
        
        st.markdown("---")
        st.markdown("##### Weather Conditions")
        c1, c2 = st.columns(2)
        with c1:
            temp = st.number_input("Temperature (°F)", value=50.0, step=1.0)
            precip = st.number_input("Precipitation (in)", value=0.0, step=0.1)
            snowfall = st.number_input("Snowfall (in)", value=0.0, step=0.1)
        with c2:
            snow_depth = st.number_input("Snow Depth (in)", value=0.0, step=0.1)
            wind_speed = st.number_input("Wind Speed (mph)", value=5.0, step=1.0)
        
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
                    "WT08": 0 # Not directly mapped from UI currently
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
                st.info("This prediction is generated using a two-stage model. Traffic volume is first estimated based on time, location, and weather, and then used to predict crash counts.")
                
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
            start_date = datetime.date.today() - datetime.timedelta(days=30)
            end_date = datetime.date.today()
            date_range = st.date_input("Date Range", (start_date, end_date))
            
            # 2. Time Period
            time_period_filter = st.selectbox("Time Period", options=["All", "Peak", "Off-Peak"])
            
            # 3. Weather
            weather_filter = st.selectbox("Weather", options=["All", "Rain", "Snow", "Fog", "Clear"])
            
            # 4. Zipcode
            zip_filter = st.text_input("Zipcode", value="", placeholder="e.g. 10001 (leave blank for All)", max_chars=5)
            
            st.markdown("<br>", unsafe_allow_html=True)
            apply_filters = st.button("Apply Filters", type="primary", use_container_width=True)

    with exp_col2:
        st.subheader("Visual Analysis")
        
        # Load data once
        try:
            full_df = get_raw_data()
            if apply_filters:
                # Note: Date range is kept in UI but ignored in filter_data for now as DATE column is missing
                df_to_plot = filter_data(full_df, time_period_filter, weather_filter, zip_filter)
                st.toast(f"Filters applied! Showing {len(df_to_plot):,} records.")
            else:
                df_to_plot = full_df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            df_to_plot = pd.DataFrame()

        # 1. Crash Density by ZIP Code
        with st.container(border=True):
            st.markdown("#### Average Crash Density (Engineered)")
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
            st.markdown("#### Traffic Volume vs. Crash Count")
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
                st.caption("Correlation between estimated traffic volume and crash frequency. Blue: Off-Peak, Red: Peak.")
            else:
                st.warning("No data available for correlation chart.")
