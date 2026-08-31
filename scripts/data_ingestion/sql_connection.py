# %% libraries and packages
from pathlib import Path
import pandas as pd
import pyodbc

# %% load data demographic.csv
BASE_DIR = Path(__file__).resolve().parents[2]
FILE_NAME = "demographic.csv"
DATA_PATH = BASE_DIR / "data" / "processed" / FILE_NAME
df = pd.read_csv(DATA_PATH)
df



# %% load data location.csv 
BASE_DIR = Path(__file__).resolve().parents[2]
FILE_NAME = "location.csv"
DATA_PATH = BASE_DIR / "data" / "processed" / FILE_NAME
df = pd.read_csv(DATA_PATH)
df


# %% load data account.csv 
BASE_DIR = Path(__file__).resolve().parents[2]
FILE_NAME = "account.csv"
DATA_PATH = BASE_DIR / "data" / "processed" / FILE_NAME
df = pd.read_csv(DATA_PATH)
df




# %% Create Connection
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=.\\SQLEXPRESS;"
    "Database=BankChurn;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()

print("SQL Server connection successful!")




# %% Push demographic to database
cursor.execute("SET IDENTITY_INSERT demographic ON")

for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO demographic (
            CustomerId,
            Gender,
            Age,
            Salary,
            LocationId,
            Churned
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """,
    int(row.CustomerId),
    row.Gender,
    int(row.Age),
    float(row.Salary),
    int(row.LocationId),
    int(row.Churned)
    )
cursor.execute("SET IDENTITY_INSERT demographic OFF")
conn.commit()
print("Data inserted successfully")




# %% Push location to database
cursor.execute("SET IDENTITY_INSERT location ON")

for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO location (
            LocationId,
            Geography
        )
        VALUES (?, ?)
    """,
    int(row.LocationId),
    row.Geography
    )
cursor.execute("SET IDENTITY_INSERT location OFF")
conn.commit()
print("Data inserted successfully")




# %% Push Account to Database

for index, row in df.iterrows():

    cursor.execute("""
        INSERT INTO account (
            CustomerId,
            Tenure,
            Balance,
            NumProducts,
            HasCreditCard,
            IsActive
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        int(row.CustomerId),
        int(row.Tenure),
        float(row.Balance),
        int(row.NumProducts),
        int(row.HasCreditCard),
        int(row.IsActive)
    )

conn.commit()

print("Account data inserted successfully")
