import pandas as pd
from joblib import load as ld
from sklearn.preprocessing import StandardScaler

test_data = pd.read_csv("Machine_Learning-Mary-Kavalionak\\Lab\\Data\\test_samples.csv", index_col= "id")

model = ld('Machine_Learning-Mary-Kavalionak\\Lab\\model_lgr.pkl')

X_test = test_data.drop("cardio", axis = 1)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X_test)

probabilities = model.predict_proba(scaled_X)

predictions = model.predict(scaled_X)

results = pd.DataFrame({
    'probability class 0': probabilities[:, 0],
    'probability class 1': probabilities[:, 1],
    'prediction': predictions
})

results.to_csv("Machine_Learning-Mary-Kavalionak\\Lab\\predictions.csv")