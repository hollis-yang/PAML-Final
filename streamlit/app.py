import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt

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
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        padding: 0.8rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
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
                # Placeholder prediction logic
                predicted_crashes = 42
                estimated_volume = 15230
                
                # Prominent Predicted Crash Count
                st.markdown(f"<h1 style='text-align: center; color: #ff4b4b; margin-bottom: 0px;'>Predicted Crash Count: {predicted_crashes}</h1>", unsafe_allow_html=True)
                # Smaller Estimated Traffic Volume
                st.markdown(f"<h4 style='text-align: center; color: gray; margin-top: 0px;'>Estimated Traffic Volume: {estimated_volume:,} vehicles</h4>", unsafe_allow_html=True)
                
                # Model explanation
                st.info("This prediction is generated using a two-stage model. Traffic volume is first estimated based on time, location, and weather, and then used to predict crash counts.")
                
                # Input summary box
                st.markdown("---")
                st.markdown("##### Input Summary")
                st.write(f"**ZIP Code:** {zip_code if zip_code else 'Not specified'}  |  **Date:** {date_input}  |  **Time:** {time_period}")
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
                st.markdown("##### Historical Crash Counts (2020–2025)")
                
                # Generate placeholder historical data including the prediction
                years = list(range(2020, 2026))
                hist_counts = [35, 38, 40, 37, 45, predicted_crashes]
                
                chart_data = pd.DataFrame({
                    "Year": years,
                    "Crash Count": hist_counts,
                    "Type": ["Historical"] * 5 + ["Predicted"]
                })
                
                # Create Altair chart for better control over the highlighted predicted point
                base = alt.Chart(chart_data).encode(
                    x=alt.X('Year:O', axis=alt.Axis(labelAngle=0))
                )
                line = base.mark_line(color='#1f77b4').encode(
                    y='Crash Count:Q'
                )
                points = base.mark_circle(size=100).encode(
                    y='Crash Count:Q',
                    color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Predicted'], range=['#1f77b4', '#ff4b4b']), legend=alt.Legend(title="Data Type"))
                )
                
                # Label the predicted point
                pred_data = chart_data[chart_data['Type'] == 'Predicted']
                text = alt.Chart(pred_data).mark_text(
                    align='left', baseline='middle', dx=10, dy=-15, fontSize=14, color='#ff4b4b', text="Pred."
                ).encode(
                    x=alt.X('Year:O'), 
                    y='Crash Count:Q'
                )
                
                st.altair_chart(line + points + text, use_container_width=True)
                
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
        
        if apply_filters:
            st.toast("Filters applied!", icon="✅")
        
        # 1. Crash Density by ZIP Code
        with st.container(border=True):
            st.markdown("#### Crash Density by ZIP Code")
            # Generate placeholder map data around NYC
            map_data = pd.DataFrame(
                np.random.randn(150, 2) / [50, 50] + [40.7128, -74.0060],
                columns=['lat', 'lon']
            )
            import plotly.express as px
            fig = px.scatter_mapbox(map_data, lat="lat", lon="lon", zoom=9, height=350, opacity=0.6)
            fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Highlights spatial differences in crash frequency across NYC.")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # 2. Crash Heatmap by Hour & Day
        with st.container(border=True):
            st.markdown("#### Crash Heatmap by Hour & Day")
            
            # Generate placeholder heatmap data
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            hours = list(range(24))
            heatmap_data = []
            for d in days:
                for h in hours:
                    # make it peak at 8 and 17
                    base = 10
                    if 7 <= h <= 9 or 16 <= h <= 18:
                        base = 50
                    count = np.random.poisson(base)
                    heatmap_data.append({"Day of Week": d, "Hour of Day": h, "Crashes": count})
            df_heatmap = pd.DataFrame(heatmap_data)
            
            heat = alt.Chart(df_heatmap).mark_rect().encode(
                x=alt.X('Day of Week:O', sort=days),
                y=alt.Y('Hour of Day:O'),
                color=alt.Color('Crashes:Q', scale=alt.Scale(scheme='orangered'))
            ).properties(height=350)
            
            st.altair_chart(heat, use_container_width=True)
            st.caption("Shows peak crash periods across different days and hours.")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # 3. Crash Count Distribution by ZIP Code
        with st.container(border=True):
            st.markdown("#### Crash Count Distribution by ZIP Code")
            
            zips = ["10001", "10002", "11201", "11101", "10451", "11234", "10013", "10306", "11373", "11207"]
            counts = np.random.randint(50, 500, size=len(zips))
            df_dist = pd.DataFrame({"ZIP Code": zips, "Crash Count": counts}).sort_values("Crash Count", ascending=False)
            
            bar = alt.Chart(df_dist).mark_bar(color='#1f77b4').encode(
                x=alt.X('ZIP Code:N', sort='-y', axis=alt.Axis(labelAngle=-45)),
                y='Crash Count:Q'
            ).properties(height=350)
            
            st.altair_chart(bar, use_container_width=True)
            st.caption("Displays the distribution of crash counts across ZIP codes.")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # 4. Traffic Volume vs. Crash Count
        with st.container(border=True):
            st.markdown("#### Traffic Volume vs. Crash Count")
            
            df_scatter = pd.DataFrame({
                "Traffic Volume": np.random.randint(1000, 20000, 200),
                "Noise": np.random.normal(0, 20, 200)
            })
            # Generate positive correlation
            df_scatter["Crash Count"] = df_scatter["Traffic Volume"] * 0.005 + df_scatter["Noise"] + 10
            df_scatter["Crash Count"] = df_scatter["Crash Count"].clip(lower=0)
            
            scatter = alt.Chart(df_scatter).mark_circle(size=60, opacity=0.6, color='#ff4b4b').encode(
                x=alt.X('Traffic Volume:Q', title='Traffic Volume', scale=alt.Scale(zero=False)),
                y=alt.Y('Crash Count:Q', title='Crash Count')
            ).properties(height=350)
            
            st.altair_chart(scatter.interactive(), use_container_width=True)
            st.caption("Illustrates the relationship between traffic volume and crash count.")
