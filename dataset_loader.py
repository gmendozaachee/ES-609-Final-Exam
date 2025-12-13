import pandas as pd
DEVICE_DATA_PATH = r"C:\Users\gmendozaachee\OneDrive\Documents\ES A603\4_Data\Home Use Medical Devices Condensed.xlsx"
ANCHORAGE_DATA_PATH = r"C:\Users\gmendozaachee\Documents\cvx\Thesis\Medical Loads Project\4_Data\Anchorage_Complied_Data.xlsx"

def load_devices():
    df = pd.read_excel(DEVICE_DATA_PATH, engine="openpyxl")

    device_brands = {}
    device_power  = {}

    for _, row in df.iterrows():
        device = str(row[0]).strip()
        brand  = str(row[1]).strip()
        power  = row[3]

        device_brands.setdefault(device, []).append(brand)
        device_power[(device, brand)] = power

    return df, device_brands, device_power

def load_anchorage_data(start_time=None, end_time=None):
    df = pd.read_excel(ANCHORAGE_DATA_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    if start_time and end_time:
        df_24 = df.loc[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]
    else:
        df_24 = df

    temperature = df_24['Ambient Temperature (C)'].values
    solar_kw    = df_24['AC System Output (W)'].values / 1000.0
    timestamps  = df_24['Timestamp'].values
    return df_24, temperature, solar_kw, timestamps


