from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
"""
#testing
new_model = load_model('first_model.h5')

a = np.array([12,36,2,56])
print(a.shape)

scaler = MinMaxScaler(feature_range=(0,1))
sA = scaler.fit_transform(a.reshape(1,-1))

prediction = new_model.predict_classes(sA, verbose=1)
print(prediction)
"""

path = 'test_y'

df = pd.read_csv(path)

values = df.values

j = 0
for x in range(len(values)):
    if (values[x] == 1):
        j += 1

per_0 = j / len(values)

print(per_0)