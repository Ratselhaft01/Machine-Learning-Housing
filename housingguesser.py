import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Getting my Dataset loaded into my code and 
dataset = pd.read_csv('Housing.csv')
dataset2 = dataset.drop(['mainroad', 'guestroom', 'basement', 'hotwaterheating'],axis='columns')
X = dataset2.iloc[:, 1:9].values
y = dataset2.iloc[:, 0].values

# Getting sklearn's labelencoder to change the strings into something the model can understand
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 4] = le.fit_transform(X[:, 4])
X[:, 6] = le.fit_transform(X[:, 6])
X[:, 7] = le.fit_transform(X[:, 7])
# print(X)

# Splitting the dataset into training and testing, so we can test how accurate our model is
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Scaling the training and testing sets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# The actual model itself. It has 5 layers.
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(128, activation='relu'))
ann.add(tf.keras.layers.Dense(64, activation='relu'))
ann.add(tf.keras.layers.Dense(32, activation='relu'))
ann.add(tf.keras.layers.Dense(612, activation='relu'))
ann.add(tf.keras.layers.Dense(1))

# Compiling the model with an optimizer (so it knows how to get better), 
# and the loss (so it knows how bad it did).
ann.compile(optimizer = 'adam', loss = 'mse')

# This is where we actually do the training. We give it the data and tell it how
# much data to take at once, and how long to train for.
ann.fit(X_train, y_train, batch_size = 50, epochs = 100)

# Our first prediction using the test set to see how accurate the model is.
result = ann.predict(X_test)
result = pd.DataFrame(result,columns=['Price'])
print(result.head())
print(y_test[0:5])

# Saving the model, which works.
ann.save("model.h5")

# Loading the model, but gives a weird "hasn't been fit yet" error.
model = keras.models.load_model('model.h5')

# A function to prepare the new prediction data and then make a prediction with it.
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

# The code that keeps getting values from the user to make predictions
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