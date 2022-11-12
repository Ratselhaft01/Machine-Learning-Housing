import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Getting my Dataset loaded into my code and 
dataset = pd.read_csv('Housing.csv')
dataset2 = dataset.drop(['mainroad', 'guestroom', 'basement', 'hotwaterheating'],axis='columns')
X = dataset2.iloc[:, 1:9].values
y = dataset2.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 4] = le.fit_transform(X[:, 4])
X[:, 6] = le.fit_transform(X[:, 6])
X[:, 7] = le.fit_transform(X[:, 7])
# print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(128, activation='relu'))
ann.add(tf.keras.layers.Dense(64, activation='relu'))
ann.add(tf.keras.layers.Dense(32, activation='relu'))
ann.add(tf.keras.layers.Dense(612, activation='relu'))
ann.add(tf.keras.layers.Dense(1))

ann.compile(optimizer = 'adam', loss = 'mse')

ann.fit(X_train, y_train, batch_size = 50, epochs = 100)

result = ann.predict(X_test) # Prediction using model
result = pd.DataFrame(result,columns=['Price']) # Dataframe
print(result.head())
print(y_test[0:5])

ann.save("model.h5")

model = keras.models.load_model('model.h5')

def predict_price(area, bedrooms, bathroom, stories, airc, parking, prefarea, furnished):
    if airc == 'no': airc = 0 
    else: airc = 1
    if prefarea == 'no': prefarea = 0 
    else: prefarea = 1
    if furnished == 'furnished': furnished = 0 
    elif furnished == 'semi-furnished': furnished = 1
    else: furnished = 2
    
    x = [area, bedrooms, bathroom, stories, airc, parking, prefarea, furnished]
    x = sc.transform([x])
    return ann.predict(x)[0][0]

print("\n\n\n\n\nTo get a prediction on your housing price, enter values like this.")
print((7420, 4, 2, 3, 'yes', 2, 'yes', 'furnished'))

predict = 'yes'
while predict == 'yes':
    area = int(input("\nSquare Footage of the house: "))
    bedrooms = int(input("Bedrooms in the house: "))
    bathroom = int(input("Bathrooms in the house: "))
    stories = int(input("Stories in the house: "))
    airc = input("Airconditioning in the house (yes/no): ").lower()
    parking = int(input("Parking: "))
    prefarea = input("Is it a Preferred Area? ").lower()
    furnished = input("Furnished (furnished/semi-furnished/unfurnished) ").lower()
    print(predict_price(area, bedrooms, bathroom, stories, airc, parking, prefarea, furnished))

    predict = input("Predict another price? (yes/no): ").lower()