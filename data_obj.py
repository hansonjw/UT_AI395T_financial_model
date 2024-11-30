import requests
import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from pandas.tseries.offsets import DateOffset


class DataObj():

    def __init__(self):
        self.df = None
        self.create_data_frame()


    def create_data_frame(self):
        df_snp = SandP().df
        df_other = MacroData().df
        df = pd.concat([df_snp, df_other], axis=1)

        for col in df.columns:
            if col not in ['Close', 'Close Norm', 'Daily Delta', '10 yr', 'CAGR 10 yr', '5 yr', 'CAGR 5 yr',
                        'FF_Rate', 'AAA_Rate', 'Fed_Receipts']:
                df[col] = df[col].interpolate(method='linear', limit_area='inside')
            elif col in ['FF_Rate', 'AAA_Rate', 'Fed_Receipts']:
                df[col] = df[col].ffill()
            else:
                pass

        self.df = df
        return

    def make_csv_file(self, file_path="./data_output/", file_name="didthiswork.csv"):
        self.df.to_csv(f"{file_path}{file_name}")
        return


class SandP():

    def __init__(self):
        self.df = None
        self.create_data_frame()

    def create_data_frame(self):

        # Call Yahoo Finance and shape the dataframe
        snp = yf.Ticker("^GSPC").history(period="max", interval="1d")
        snp.drop(labels=["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis=1, inplace=True)
        snp.reset_index(inplace=True)

        # re-index with daily frequency
        snp['Date'] = pd.to_datetime(snp['Date'].copy()).dt.date
        snp.set_index('Date', inplace=True)
        
        # make a new date axis and set new index with no date gaps
        time_axis = pd.date_range(start=min(snp.index), end=max(snp.index), freq='d')
        snp2 = snp.reindex(time_axis)
        snp3 = snp2.ffill(axis=0)

        # Add 5 and 10 yr compound growth rates
        snp3["Close Norm"] = snp3["Close"]/snp3["Close"].iloc[0]
        snp3["Daily Delta"] = 100*(snp3["Close"]/snp3["Close"].shift(periods=+1) - 1)
        snp3['10 yr'] = snp3['Close Norm'].shift(periods=+3652)
        snp3['CAGR 10 yr'] = (snp3['Close Norm']/snp3['10 yr'])**(1/10)-1
        snp3['5 yr'] = snp3['Close Norm'].shift(periods=int(round(+3652/2,0)))
        snp3['CAGR 5 yr'] = (snp3['Close Norm']/snp3['5 yr'])**(1/5)-1

        self.df = snp3
        return

    def write_df_to_file(self):
        self.df.to_csv("snp.csv")


class MacroData:

    def __init__(self):
        self.df = None
        self.df_list = []

        # Place fed key here to get this model to work
        self.fred_key = ''
        self.fred_series = [
            {'fred_id': 'GDP', 'desc': 'US GDP', 'freq': 'Quarter', 'Name': 'US_GDP'},
            {'fred_id': 'UNRATE', 'desc': 'US Unemployement Rate', 'freq': 'Month', 'Name': 'US_Uemp'},
            {'fred_id': 'FEDFUNDS', 'desc': 'US Federal Funds Rate', 'freq': 'Month', 'Name': 'FF_Rate'},
            {'fred_id': 'AAA', 'desc': 'US AAA Average Corporate Bond Rate', 'freq': 'Month', 'Name': 'AAA_Rate'},
            {'fred_id': 'B230RC0Q173SBEA', 'desc': 'US Population', 'freq': 'Quarter', 'Name': 'US_Pop'},
            {'fred_id': 'CPIAUCNS', 'desc': 'US Inflation, CPI', 'freq': 'Month', 'Name': 'US_CPI'},
            {'fred_id': 'FYFR', 'desc': 'Federal Receipts', 'freq': 'Year', 'Name': 'Fed_Reciepts'},
            {'fred_id': 'FYGFD', 'desc': 'Federal Gross Debt', 'freq': 'Year', 'Name': 'Fed_Debt'},
            {'fred_id': 'M2NS', 'desc': 'US M2 Money Supply', 'freq': 'Month', 'Name': 'US_M2'}
        ]

        self._load_house_data()
        self._load_energy_data()
        self._load_gold_data()
        self._load_population_data()        
        self._load_fred_data()
        
        self._concat_df_list()


    def call_fred(self, fred_id):
        res = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={fred_id}&api_key={self.fred_key}&file_type=json")
        df = pd.DataFrame.from_records(res.json()['observations'])
        return df

    def _load_fred_data(self):
        for x in self.fred_series:
            df0 = self.call_fred(x['fred_id'])
            df1 = df0[['date','value']].copy()
            df1['value'] = pd.to_numeric(df1['value'], errors="coerce")
            df1['Date'] = pd.to_datetime(df1['date']).dt.date
            df2 = df1.rename(columns={'value': x['Name']})
            df2.drop('date', axis=1, inplace=True)
            df3 = df2.set_index('Date')
            self.df_list.append(df3)

    def _concat_df_list(self):
        self.df = pd.concat(self.df_list, axis=1)
        self.df.apply(pd.to_numeric, errors='coerce')
        self.df.sort_index(inplace=True)
        self.df.rename({"London Market Price (British &pound; [1718-1949] or U.S. $ [1950-2011] per fine ounce)": "Gold_London",
           "Gold/Silver Price Ratio (ounces of silver per ounce of gold)": "GS_Ratio",
           "New York Market Price (U.S. dollars per fine ounce)": "Gold_NY",
           "ENERGY (TWh)":"W_Energy",
           "POPULATION":"W_Pop"},
           axis=1, inplace=True)
        self.df.index = pd.to_datetime(self.df.index)
        return

    def _load_house_data(self):
        df = pd.read_csv("data_features/house.csv")
        df.rename(columns={"Composite Index": "House"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Composite Date"]).dt.date
        df2 = df[["Date", "House"]]
        df2.set_index("Date", inplace=True)
        self.df_list.append(df2)
        return

    def _load_energy_data(self):
        df = pd.read_csv("data_features/global-energy.csv")
        df["Date"] = pd.to_datetime(df["DATE"]).dt.date
        df2 = df[["ENERGY (TWh)","Date"]]
        df2.set_index("Date", inplace=True)
        self.df_list.append(df2)
        return
    
    def _load_gold_data(self):
        df = pd.read_csv("data_features/gold.csv")
        df["day"] = 1
        df["month"] = 1
        df.rename(columns={"Year": "year"}, inplace=True)
        df["Date"] = pd.to_datetime(df[["month", "day", "year"]]).dt.date
        df.set_index("Date", inplace=True)
        df.drop(["month", "day", "year"], axis=1, inplace=True)
        self.df_list.append(df)
        return

    def _load_population_data(self):
        df = pd.read_csv("data_features/population.csv")
        df["Date"] = pd.to_datetime(df["DATE"]).dt.date
        df.set_index("Date", inplace=True)
        df2 = df[["POPULATION"]]
        self.df_list.append(df2)
        return