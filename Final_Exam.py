import streamlit as st
import pandas as pd
import numpy as np
from dataset_loader import load_devices, load_anchorage_data
from PIL import Image
import os
import cvxpy as cp
import streamlit as st
##################################################################################################################################################

##################################################################################################################################################



st.set_page_config(
    page_title="UAA Energy Assistant",
    page_icon="⚡",
    layout="wide"
)
##################################################################################################################################################

# banner_path = r"C:\Users\gmendozaachee\OneDrive\Documents\ES A603\Final_Exam\uaa banner.png"

# if os.path.exists(banner_path):
#     banner = Image.open(banner_path)
#     MAX_HEIGHT = 200 

#     w, h = banner.size
#     scale = MAX_HEIGHT / h
#     banner_resized = banner.resize((int(w * scale), MAX_HEIGHT))

#     st.image(banner_resized, use_container_width=False)
# else:
#     st.warning("⚠ UAA banner image not found.")

# banner_path = r"C:\Users\gmendozaachee\OneDrive\Documents\ES A603\Final_Exam\uaa banner.png"
banner_path = "uaa banner.png"


if os.path.exists(banner_path):
    banner = Image.open(banner_path)

    MAX_HEIGHT = 400  # adjust if needed
    w, h = banner.size
    scale = MAX_HEIGHT / h
    banner_resized = banner.resize((int(w * scale), MAX_HEIGHT))

    col_left, col_center, col_right = st.columns([5,5,5])
    with col_center:
        st.image(banner_resized)

else:
    st.warning("⚠ UAA banner image not found.")


##################################################################################################################################################

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

T = 24

hour_labels = [
    "9 AM", "10 AM", "11 AM", "12 PM",
    "1 PM", "2 PM", "3 PM", "4 PM",
    "5 PM", "6 PM", "7 PM", "8 PM",
    "9 PM", "10 PM", "11 PM", "12 AM",
    "1 AM", "2 AM", "3 AM", "4 AM",
    "5 AM", "6 AM", "7 AM", "8 AM"
]

UAA_GREEN = "#00583d"
UAA_GOLD  = "#ffc425"

##################################################################################################################################################

# import streamlit as st
# import pandas as pd
# import numpy as np
# from dataset_loader import load_devices, load_anchorage_data
# from PIL import Image

# banner_path = r"C:\Users\gmendozaachee\OneDrive\Documents\ES A603\Final_Exam\uaa banner.png"
# banner = Image.open(banner_path)
# st.image(banner, use_container_width=True)


# st.markdown(
#     """
#     <style>
#     html, body, [class*="css"]  {
#         font-size: 22px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# T = 24

# hour_labels = [
#     "9 AM", "10 AM", "11 AM", "12 PM",
#     "1 PM", "2 PM", "3 PM", "4 PM",
#     "5 PM", "6 PM", "7 PM", "8 PM",
#     "9 PM", "10 PM", "11 PM", "12 AM",
#     "1 AM", "2 AM", "3 AM", "4 AM",
#     "5 AM", "6 AM", "7 AM", "8 AM"
# ]


# UAA_GREEN = "#00583d"
# UAA_GOLD = "#ffc425"

# st.set_page_config(
#     page_title="UAA Energy Assistant",
#     page_icon="⚡",
#     layout="wide"
# )

##################################################################################################################################################

df_devices, device_brands, device_power = load_devices()
df_env, temperature_full, solar_kw_full, timestamps_full = load_anchorage_data()

# st.success("Datasets loaded successfully!")
# st.subheader("Medical Device Dataset")
# st.dataframe(df_devices)

# st.subheader("Anchorage Environmental Dataset (first 10 rows)")
# st.dataframe(df_env.head(10))

st.session_state.temperature_full = temperature_full
st.session_state.solar_kw_full = solar_kw_full
st.session_state.timestamps_full = timestamps_full

available_dates = sorted(df_env["Timestamp"].dt.date.unique())
st.session_state.setdefault("available_dates", available_dates)
st.session_state.setdefault("selected_date", available_dates[0])

st.session_state.setdefault("final_devices", [])
st.session_state.setdefault("final_brands", [])
st.session_state.setdefault("final_power", [])
st.session_state.setdefault("final_vectors", [])

defaults = {
    "device_brands": device_brands,
    "device_power": device_power,
    "page": "inputs",
    "num_devices": 1,
    "min_temp": 69.0,
    "max_temp": 72.0,
    "batt_cap": 13.5,
    "ev_cap": 75.0,
    "ev_init": 10.0,
    "bess_init": 5.0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

if "last_num_devices" not in st.session_state:
    st.session_state.last_num_devices = st.session_state.num_devices

if st.session_state.last_num_devices != st.session_state.num_devices:
    for i in range(20):
        for hr in range(24):
            key = f"v_{i}_{hr}"
            if key in st.session_state:
                del st.session_state[key]
    st.session_state.last_num_devices = st.session_state.num_devices

st.markdown(
    f"""
    <div style="
        background-color:{UAA_GREEN};
        padding:15px;
        border-radius:8px;
        text-align:center;">
        <h1 style="color:{UAA_GOLD}; margin-bottom:5px;">
            UAA Smart Home Energy Input
        </h1>
        <p style="color:white; font-size:14px;">
            Enter system inputs to begin your analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

if st.session_state.page == "inputs":

    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color:{UAA_GOLD};
                padding:20px;
                border-radius:10px;
                color:{UAA_GREEN};
                font-size:16px;">
            """,
            unsafe_allow_html=True
        )

        # st.session_state.selected_date = st.selectbox(
        #     "Select simulation date (Anchorage dataset):",
        #     options=st.session_state.available_dates,
        #     index=st.session_state.available_dates.index(st.session_state.selected_date),
        # )
        
        st.session_state.selected_date = pd.to_datetime("2024-01-01").date()

        st.session_state.num_devices = st.number_input(
            "How many devices?",
            min_value=1,
            max_value=3,
            value=st.session_state.num_devices,
            step=1,
        )

        st.session_state.min_temp = st.number_input(
            "Minimum Temperature (°F):", value=st.session_state.min_temp
        )

        st.session_state.max_temp = st.number_input(
            "Maximum Temperature (°F):", value=st.session_state.max_temp
        )

        st.session_state.batt_cap = st.number_input(
            "Battery Capacity (kWh):", value=st.session_state.batt_cap
        )

        st.session_state.ev_cap = st.number_input(
            "EV Capacity (kWh):", value=st.session_state.ev_cap
        )

        st.session_state.ev_init = st.number_input(
            "EV Initial Energy (kWh):", value=st.session_state.ev_init
        )

        st.session_state.bess_init = st.number_input(
            "BESS Initial Energy (kWh):", value=st.session_state.bess_init
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    if st.button("Continue → Device Selection"):
        st.session_state.page = "devices"
        st.rerun()

if st.session_state.page == "devices":

    st.markdown(
        f"<h2 style='color:{UAA_GREEN};'>Device Selection</h2>",
        unsafe_allow_html=True
    )

    device_brands = st.session_state.device_brands
    device_power = st.session_state.device_power
    num_devices = st.session_state.num_devices

    selected_devices = []
    selected_brands = []
    selected_power = []
    vectors = []
###################################################################################################################################################################################

    for i in range(num_devices):

        st.write(f"### Device #{i+1}")

        device = st.selectbox(
            "Select Device:",
            [
                d for d in device_brands.keys()
                if d not in selected_devices
            ],
            key=f"dev_{i}"
        )

        brand_list = device_brands.get(device, [])
        brand = st.selectbox(
            "Select Brand:",
            brand_list,
            key=f"br_{i}"
        )

        power = device_power.get((device, brand), None)
        if power is None:
            st.warning("⚠ Power rating not found — this device cannot be optimized.")
        else:
            st.info(f"Power Requirement: **{power} W**")

        selected_devices.append(device)
        selected_brands.append(brand)
        selected_power.append(power)

        if device.lower() == "dialysis machine":
           st.warning(
         "Each hemodialysis treatment requires **5 consecutive hours** of operation.\n\n"
         "Please ensure that **start times for hemodialysis sessions are at least "
         "5 hours apart**, as overlapping sessions may result in infeasible schedules."
    )
        st.write("**24-Hour ON/OFF Pattern**")

        col1, col2, col3 = st.columns(3)

        vector = []

###################################################################################################################################################################################
        # for hour in range(24):
        #     col = [col1, col2, col3][hour % 3]
        #     state = col.checkbox(f"{hour}:00", key=f"v_{i}_{hour}")
        #     vector.append(1 if state else 0)

        for hour in range(24):
            col = [col1, col2, col3][hour % 3]
            state = col.checkbox(hour_labels[hour], key=f"v_{i}_{hour}")
            vector.append(1 if state else 0)

###################################################################################################################################################################################
        vectors.append(vector)
        st.write("---")

    if st.button("Finalize Inputs"):
        st.session_state.final_devices = selected_devices
        st.session_state.final_brands = selected_brands
        st.session_state.final_power = selected_power
        st.session_state.final_vectors = vectors

        st.session_state.page = "summary"
        st.rerun()

if st.session_state.page == "summary":

    st.markdown(f"<h2 style='color:{UAA_GREEN};'>Summary of Inputs</h2>",
                unsafe_allow_html=True)

    st.success("All device selections have been saved successfully.")

    if st.button("Go Back"):
        st.session_state.page = "devices"
        st.rerun()

if st.session_state.page == "summary":

    start_time = pd.Timestamp(st.session_state.selected_date) + pd.Timedelta(hours=9)
    end_time   = start_time + pd.Timedelta(hours=23)

    df_24, temperature_24, solar_kw_24, timestamps_24 = load_anchorage_data(
        start_time=start_time, end_time=end_time
    )

    st.session_state.temperature_24 = temperature_24
    st.session_state.solar_24 = solar_kw_24
    st.session_state.timestamps_24 = timestamps_24

    T = 24
    solar_power_available = solar_kw_24

    E_batt_max = st.session_state.batt_cap
    E_batt_min = 0.0
    batt_ch_max = 5.0
    batt_dis_max = 5.0
    eta_batt_ch = 0.90
    eta_batt_dis = 0.90
    E_batt_init = st.session_state.bess_init

    bill = 248.82
    price = 0.4601
    monthly_kwh = bill / price
    daily_kwh = monthly_kwh / 30
    hourly_kwh = daily_kwh / 24
    base_load_demand = 0.8 * hourly_kwh * np.ones(T)

    delta_t = 1.0
    house_area = 120.0
    Rth = 1 / (0.00075 * house_area)
    Cth = 0.005 * house_area
    a_temp = 1 - delta_t / (Cth * Rth)
    b_temp = Rth

    def COP_3(Tc):
        return np.maximum(1.2, 4 - 0.07 * (20 - Tc))

    ev_capacity = st.session_state.ev_cap
    ev_init = st.session_state.ev_init
    ev_dis_max = 10
    eta_ev_dis = 0.90
    outage_hours = np.arange(0, 24)

    Tcomfort_C = (st.session_state.max_temp - 32) * 5 / 9
    Tmin_C     = (st.session_state.min_temp - 32) * 5 / 9
    Tmax_C     = Tcomfort_C

    devices  = st.session_state.final_devices
    power    = st.session_state.final_power
    vectors  = st.session_state.final_vectors

    expected_humidifier   = np.zeros(T)
    expected_oxygen       = np.zeros(T)
    expected_hemodialysis = np.zeros(T)
    
    dialysis_starts = []

    for i, dev in enumerate(devices):
        if dev.lower() == "humidifier":
            expected_humidifier = (power[i] / 1000.0) * np.array(vectors[i])
            break

    for i, dev in enumerate(devices):
        if dev.lower() == "oxygen concentrator":
            expected_oxygen = (power[i] / 1000.0) * np.array(vectors[i])
            break

    dialysis_starts = []
    dialysis_disinfect_kw = 0.61
    dialysis_treatment_kw = 0.0

    for i, dev in enumerate(devices):
        if dev.lower() == "dialysis machine":
            v = np.array(vectors[i])
            dialysis_starts = list(np.where(v == 1)[0])
            dialysis_treatment_kw = power[i] / 1000.0
            break

    if dialysis_starts:
        for start in dialysis_starts:
            disinfect = (start - 1) % 24
            expected_hemodialysis[disinfect] = dialysis_disinfect_kw
            for h in [(disinfect + x) % 24 for x in range(1, 5)]:
                expected_hemodialysis[h] = dialysis_treatment_kw

    st.session_state.expected_humidifier     = expected_humidifier
    st.session_state.expected_oxygen         = expected_oxygen
    st.session_state.expected_hemodialysis   = expected_hemodialysis
    st.session_state.base_load_demand        = base_load_demand

    oxygen_usage_schedule = (expected_oxygen > 0).astype(int)
    oxygen_power_kw = np.max(expected_oxygen) if np.max(expected_oxygen) > 0 else 0.0

    if len(dialysis_starts) > 0:
     num_sessions = len(np.unique([(h+1) % 24 for h in dialysis_starts]))
    else:
     num_sessions = 0


    Min_Possible_Temp_C = Tmin_C
    Max_Possible_Temp_C = Tmax_C
    
    P_hp_min = 0.0
    P_hp_max = 2.5
####################################################################################################################################################################################

    import cvxpy as cp

    Tin = cp.Variable(T)
    X_hp = cp.Variable(T)
    E_batt = cp.Variable(T + 1)
    P_batt_ch = cp.Variable(T)
    P_batt_dis = cp.Variable(T)
    P_batt_ch_solar = cp.Variable(T)
    P_batt_ch_grid = cp.Variable(T)
    P_grid = cp.Variable(T)
    P_solar_to_load = cp.Variable(T)

    P_oxygen = cp.Variable(T)
    C_oxygen = cp.Variable(T)

    P_hemodialysis = cp.Variable(T)
    C_hemodialysis = cp.Variable(T)

    P_base_supplied = cp.Variable(T)
    C_base = cp.Variable(T)

    P_ev_dis = cp.Variable(T)
    SOC_ev = cp.Variable(T + 1)

    C_temp = cp.Variable(T)
    C_temp_low  = cp.Variable(T)
    C_temp_high = cp.Variable(T)

    P_humidifier = cp.Variable(T)
    C_humidifier = cp.Variable(T)
    y_humidifier = cp.Variable(T, boolean=True)

    z_batt = cp.Variable(T, boolean=True)
    y_batt = cp.Variable(T, boolean=True)

    y_oxy_off   = cp.Variable(T, boolean=True)
    y_oxy_300   = cp.Variable(T, boolean=True)
    y_oxy_500   = cp.Variable(T, boolean=True)
    y_oxy_1000  = cp.Variable(T, boolean=True)

    if num_sessions > 0:
        y_dialysis_start = cp.Variable(num_sessions, boolean=True)
    else:
        y_dialysis_start = cp.Variable(1, boolean=True)

    w_oxygen       = 70
    w_hemodialysis = 100
    w_humidifier   = 50
    w_temp         = 30
    w_base         = 10
    max_household_power = 50.0

    is_outage = np.isin(np.arange(T), outage_hours).astype(int)

    objective = cp.Minimize(
          w_oxygen       * cp.sum(cp.multiply(is_outage, C_oxygen))
        + w_hemodialysis * cp.sum(cp.multiply(is_outage, C_hemodialysis))
        + w_humidifier   * cp.sum(C_humidifier)
        + w_base         * cp.sum(C_base)
        + w_temp         * cp.sum(C_temp / 7)
    )

    constraints = []

    if num_sessions == 0:
        constraints += [y_dialysis_start[0] == 0]

    for t in range(T):

        if oxygen_usage_schedule[t] > 0 and expected_oxygen[t] > 0:

            constraints += [
                P_oxygen[t] == (
                    oxygen_power_kw * 0.0 * y_oxy_off[t] +
                    oxygen_power_kw * 0.3 * y_oxy_300[t] +
                    oxygen_power_kw * 0.5 * y_oxy_500[t] +
                    oxygen_power_kw * 1.0 * y_oxy_1000[t]
                ),
                y_oxy_off[t] + y_oxy_300[t] + y_oxy_500[t] + y_oxy_1000[t] == 1
            ]

            if t not in outage_hours:
                constraints += [
                    y_oxy_300[t] == 0,
                    y_oxy_500[t] == 0,
                    y_oxy_1000[t] == 1,
                    y_oxy_off[t] == 0
                ]

            constraints += [
                C_oxygen[t] == expected_oxygen[t] - P_oxygen[t],
                C_oxygen[t] >= 0
            ]

        else:
            constraints += [
                P_oxygen[t] == 0,
                C_oxygen[t] == 0,
                y_oxy_off[t] == 0,
                y_oxy_300[t] == 0,
                y_oxy_500[t] == 0,
                y_oxy_1000[t] == 0
            ]

    for t in range(T):

        P_hemo_sum = 0

        for i, start_hour in enumerate(dialysis_starts):

            disinfect = start_hour - 1
            treatment = list(range(disinfect + 1, disinfect + 5))
            session_hours = [disinfect] + treatment

            if t in session_hours:
                P_hemo_sum += y_dialysis_start[i] * expected_hemodialysis[t]
                if len(outage_hours) == 0:
                    constraints += [y_dialysis_start[i] == 1]

        constraints += [
            P_hemodialysis[t] == P_hemo_sum
        ]

        if t in outage_hours:
            constraints += [
                C_hemodialysis[t] >= expected_hemodialysis[t] - P_hemodialysis[t],
                C_hemodialysis[t] >= 0
            ]
        else:
            constraints += [C_hemodialysis[t] == 0]

    for t in range(T):

        expected = expected_humidifier[t]

        if expected > 0:

            constraints += [
                P_humidifier[t] == expected * y_humidifier[t],
                C_humidifier[t] == expected - P_humidifier[t],
                C_humidifier[t] >= 0,
                y_humidifier[t] * expected <= (
                    solar_power_available[t]
                    + batt_dis_max
                    + ev_dis_max
                )
            ]

        else:
            constraints += [
                P_humidifier[t] == 0,
                C_humidifier[t] == 0,
                y_humidifier[t] == 0
            ]

    for t in range(T):

        if t in outage_hours:
            constraints += [
                P_base_supplied[t] >= 0,
                P_base_supplied[t] <= base_load_demand[t],
                C_base[t] >= base_load_demand[t] - P_base_supplied[t]
            ]
        else:
            constraints += [
                P_base_supplied[t] == base_load_demand[t],
                C_base[t] == 0
            ]

    P_total_load = (
          P_oxygen
        + P_hemodialysis
        + P_humidifier
        + P_base_supplied
        + X_hp
    )

    constraints += [E_batt[0] == E_batt_init]

    for t in range(T):

        constraints += [
            E_batt[t]   >= E_batt_min,
            E_batt[t]   <= E_batt_max,
            E_batt[t+1] >= E_batt_min,
            E_batt[t+1] <= E_batt_max
        ]

        constraints += [
            E_batt[t + 1] == E_batt[t]
                            + eta_batt_ch * P_batt_ch[t]
                            - (1 / eta_batt_dis) * P_batt_dis[t]
        ]

        constraints += [
            P_batt_ch[t] == P_batt_ch_grid[t] + P_batt_ch_solar[t],
            P_batt_ch[t] >= 0,
            P_batt_ch_grid[t] >= 0,
            P_batt_ch_solar[t] >= 0
        ]

        constraints += [
            P_batt_ch[t] <= batt_ch_max * z_batt[t],
            P_batt_dis[t] >= 0,
            P_batt_dis[t] <= batt_dis_max * y_batt[t],
            z_batt[t] + y_batt[t] <= 1
        ]

        constraints += [
            P_solar_to_load[t] >= 0,
            P_solar_to_load[t] <= P_total_load[t],
            P_solar_to_load[t] + P_batt_ch_solar[t] <= solar_power_available[t]
        ]

        if t in outage_hours:
            constraints += [
                P_grid[t] == 0,
                P_batt_ch_grid[t] == 0
            ]
        else:
            constraints += [
                P_grid[t] >= 0,
                P_grid[t] <= max_household_power
            ]

        constraints += [
            P_grid[t] ==
            P_total_load[t]
            - P_solar_to_load[t]
            - P_batt_dis[t]
            - P_ev_dis[t]
            + P_batt_ch_grid[t]
        ]

    for t in range(T):
        constraints += [
            X_hp[t] >= P_hp_min,
            X_hp[t] <= P_hp_max
        ]

    constraints += [X_hp[T - 1] == X_hp[T - 2]]
    constraints += [Tin[0] == Tcomfort_C]

    for t in range(T - 1):
        T_a = temperature_24[t]
        constraints += [
            Tin[t + 1] ==
                a_temp * Tin[t]
                + (1 - a_temp) * T_a
                + b_temp * (COP_3(T_a) * X_hp[t])
        ]

    for t in range(T):
        constraints += [
            C_temp[t] >= Tcomfort_C - Tin[t],
            C_temp[t] >= 0,
            Tin[t] >= Min_Possible_Temp_C,
            Tin[t] <= Max_Possible_Temp_C
        ]

    constraints += [
        Tin[23] == Max_Possible_Temp_C
    ]

    constraints += [
        SOC_ev[0] == ev_init
    ]

    for t in range(T):
        max_ev_dis = ev_dis_max if t in outage_hours else 0.0

        constraints += [
            P_ev_dis[t] >= 0,
            P_ev_dis[t] <= max_ev_dis,
            SOC_ev[t+1] == SOC_ev[t] - (1 / eta_ev_dis) * P_ev_dis[t] * delta_t,
            SOC_ev[t] >= 0,
            SOC_ev[t] <= ev_capacity
        ]
        
    constraints += [
    SOC_ev[T] >= 0,
    SOC_ev[T] <= ev_capacity
]

####################################################################################################################################################################################

    with st.spinner("Solving energy optimization problem…"):
     prob = cp.Problem(objective, constraints)
     
 #  prob.solve(solver=cp.GUROBI, verbose=True)
    prob.solve(solver=cp.ECOS_BB, verbose=False)
    
####################################################################################################################################################################################
   
    P_solar_combined = np.round(P_solar_to_load.value + P_batt_ch_solar.value, 2)
    P_batt_dis_plot = np.round(P_batt_dis.value, 2)
    P_ev_plot = np.round(P_ev_dis.value, 2)
    P_demand = -np.round(P_total_load.value, 2)
    P_oxygen_plot = np.round(P_oxygen.value, 2)
    P_hemo_plot = np.round(P_hemodialysis.value, 2)
    P_humid_plot = np.round(P_humidifier.value, 2)
    P_base_plot = np.round(P_base_supplied.value, 2)
    P_hp_plot = np.round(X_hp.value, 2)
    P_batt_ch_total = np.round(P_batt_ch_solar.value + P_batt_ch_grid.value, 2)


#######################################################################################################################################################################################
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    x = np.arange(1, 25)
    fontsize = 12
    linewidth = 1.5
    markersize = 8
    fig, ax = plt.subplots(figsize=(20, 6))

    bar_positive = np.column_stack([
        P_solar_combined,
        P_batt_dis_plot,
        P_ev_plot,
        P_batt_ch_total
    ])

    colors_pos = [
        [0.95, 0.60, 0.10],
        [0.10, 0.45, 0.80],
        [0.50, 0.00, 0.50],
        [0.55, 0.27, 0.07]
    ]

    labels_pos = [
        'Solar',
        'Battery Discharge',
        'EV Discharge',
        'Battery Charging'
    ]

    bottom = np.zeros_like(x, dtype=float)
    for i in range(4):
        ax.bar(
            x, bar_positive[:, i],
            bottom=bottom,
            color=colors_pos[i],
            edgecolor='k',
            width=0.95,
            label=labels_pos[i]
        )
        bottom += bar_positive[:, i]

    ax.step(x, P_oxygen_plot, where='mid', color='k', linewidth=linewidth)
    ax.plot(x, P_oxygen_plot, '^k', markersize=markersize,
            markerfacecolor='k', label='Oxygen Concentrator')

    ax.step(x, P_hemo_plot, where='mid', color='blue', linewidth=linewidth)
    ax.plot(x, P_hemo_plot, 'db', markersize=markersize,
            markerfacecolor='blue', label='Hemodialysis')

    ax.step(x, P_base_plot, where='mid', color='red', linewidth=linewidth)
    ax.plot(x, P_base_plot, 'or', markersize=markersize,
            markerfacecolor='red', label='Base Load')

    ax.step(x, P_hp_plot, where='mid', color='green', linewidth=linewidth)
    ax.plot(x, P_hp_plot, 'sg', markersize=markersize,
            markerfacecolor='green', label='Heat Pump')

    color_1 = 'slategray'
    ax.step(x, P_humid_plot, where='mid', color=color_1, linewidth=linewidth)
    ax.plot(
        x, P_humid_plot,
        'pm',
        markersize=markersize,
        markerfacecolor=color_1,
        markeredgecolor=color_1,
        label='Humidifier'
    )

    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_ylim([-.03, 3.5])
    ax.set_xlim([0.5, 24.5])
    ax.set_ylabel('Power (kW)', fontsize=fontsize, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(
        ['9am','10am','11am','12pm','1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm',
         '9pm','10pm','11pm','12am','1am','2am','3am','4am','5am','6am','7am','8am'],
        rotation=45
    )

    ax.tick_params(labelsize=fontsize)
    ax.tick_params(which='both', direction='in')
    ax.grid(True, which='both', linestyle='-', color='gray', alpha=0.25)
    ax.minorticks_on()
    ax.set_facecolor('white')

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.65, 1.0),
        fontsize=fontsize,
        ncol=3,
        frameon=True,
        edgecolor='black',
        facecolor='white'
    )

    ax_inset = inset_axes(
        ax,
        width="60%",
        height="90%",
        bbox_to_anchor=(0.71, 0.63, 0.45, 0.35),
        bbox_transform=ax.transAxes,
        loc='upper left'
    )

    actual_Tin_F = Tin.value * 9/5 + 32
    T_expected_F = np.full(T, Tcomfort_C) * 9/5 + 32
    actual_Tin_F = np.round(actual_Tin_F, 2)
    T_expected_F = np.round(T_expected_F, 2)
    
    hours = np.arange(1, len(actual_Tin_F) + 1)

    ax_inset.step(hours, actual_Tin_F, where='post',
                  color='blue', linewidth=1.8)
    ax_inset.step(hours, T_expected_F, where='post',
                  color='orange', linewidth=1.8, linestyle='--')

    ax_inset.set_title("Indoor Temperature", fontsize=fontsize, pad=2)

    tick_positions = [1, 4, 7, 10, 13, 16, 19, 22]
    tick_labels    = ['9am','12pm','3pm','6pm','9pm','12am','3am','6am']

    ax_inset.set_xticks(tick_positions)
    ax_inset.set_xticklabels(tick_labels, fontsize=10)
    ax_inset.set_ylabel("°F", fontsize=fontsize)

    for spine in ax_inset.spines.values():
        spine.set_visible(True)

    # st.pyplot(fig, use_container_width=True)
    st.pyplot(fig)
    st.success("Optimization completed successfully!")






