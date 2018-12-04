import pandas as pd
from pandas import DataFrame
import sklearn as skl
from sklearn.linear_model import LinearRegression

test_frame = pd.read_csv('Experimental walking.csv', sep=';')
act_frame = test_frame.drop(['Pelvic Tilt', 'Pelvic Up/Down Obl',
    'Pelvic Int/Ext Rot', 'Hip Flex/Ext', 'Hip Flex/Ext (L)', 'Hip Ad/Ab',
'Hip Ad/Ab (L)', 'Hip Int/Ext Rot', 'Hip Int/Ext Rot (L)',
'Knee Flex/Ext','Knee Flex/Ext (L)', 'Ankle Dorsi/Plant',
'Ankle Dorsi/Plant (L)'],axis=1)

obs_frame = test_frame.drop(['rect_fem_r', 'rect_fem_l', 'hamstrings_r',
'hamstrings_l', 'bifemsh_r', 'bifemsh_l', 'tib_ant_l', 'gastroc_l'],
axis=1)
print(obs_frame.columns)
print(act_frame.columns)
test_reg = LinearRegression()
test_reg.fit(obs_frame.get_values(), act_frame.get_values())
print(test_reg.predict(obs_frame.head(3)))