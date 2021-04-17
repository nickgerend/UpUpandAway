# Written by: Nick Gerend, @dataoutsider
# Viz: "Up, Up and Away", enjoy!

import pandas as pd
import os
from datetime import datetime

df = pd.read_csv(os.path.dirname(__file__) + '/aircraft.csv', dtype=pd.StringDtype())

df['AIR WORTH DATE'] = df['AIR WORTH DATE'].str.strip()
df_b = df.loc[(df['AIR WORTH DATE'].notnull()) & (df['AIR WORTH DATE']!='')]

df_b['A_W_Date'] = [datetime.strptime(t, '%Y%m%d') for t in df_b['AIR WORTH DATE']]
df_b['A_W_Date_Year'] = df_b['A_W_Date'].dt.year

df_b['Aircraft'] = df_b['Aircraft'].str.strip()
df_b = df_b.loc[(df_b['Aircraft'] == 'Balloon')]

df_b['NO-SEATS'] = df_b['NO-SEATS'].str.strip()
df_b = df_b.loc[(df_b['NO-SEATS'].notnull()) & (df_b['NO-SEATS']!='')]

df_b['STATE'] = df_b['STATE'].str.strip()
df_b = df_b.loc[(df_b['STATE'].notnull()) & (df_b['STATE']!='')]

df_b.to_csv(os.path.dirname(__file__) + '/aircraft_balloons.csv', encoding='utf-8', index=False)

print('finished')