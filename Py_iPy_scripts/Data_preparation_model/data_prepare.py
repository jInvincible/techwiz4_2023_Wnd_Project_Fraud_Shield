import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from geopy.distance import great_circle
import datetime

# Load data
df = pd.read_csv("credit_card_fraud.csv")

# Create "distance" col
dis = df.loc[:, ['lat', 'long', 'merch_lat', 'merch_long']]
df['distance'] = dis.apply(lambda x: float(great_circle((x['merch_lat'], x['merch_long']),(x['lat'], x['long'])).kilometers), axis=1)

# Within 100km
def loc(distance):
    if distance < 100:
        result = "within 100km"
    else:
        result = "without 100km"
    return result

df["Detect_100km"] = df["distance"].apply(loc)

# Transform date time
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df['hour_times'] = df["trans_date_trans_time"].dt.strftime("%d-%H:%M")

# Create def divide hour by fraud frequency
def hour_label(hour):
        if (int(hour) >= 5 and int(hour) <= 20) :
            result = "5h-20h"
        else:
            result = "21h-4h"
        return result

# Create "hour_lable" col
df['hour'] = df.trans_date_trans_time.dt.hour.astype("str")
df["hour_label"] = df["hour"].apply(hour_label)

# Select hour, distance, amt, is_fraud
df_ml = df[["hour","distance","amt","is_fraud"]]

# Prepare train model
y = df_ml["is_fraud"].values.reshape(-1,1)
x = df_ml.drop(columns = ["is_fraud"])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, np.ravel(y), test_size=0.1, random_state = 42)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Export random forest model
from joblib import dump

# Save the model to a file
model_filename = "random_forest_model.joblib"
dump(rf, model_filename)