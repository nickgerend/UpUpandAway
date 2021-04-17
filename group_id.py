# Written by: Nick Gerend, @dataoutsider
# Viz: "Up, Up and Away", enjoy!

import pandas as pd
import os
df = pd.read_csv(os.path.dirname(__file__) + '/aircraft_balloons.csv')
df['group_id'] = -1
df_group = df.groupby(['STATE'])
for name, rows in df_group:
    id = 0
    for index, row in rows.iterrows():
        df.at[index, 'group_id'] = id
        id += 1
df.to_csv(os.path.dirname(__file__) + '/aircraft_balloons_id.csv', encoding='utf-8', index=False)

print('finished')