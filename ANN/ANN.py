from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras import Sequential
from keras import layers
from keras import models
dataset = datasets.load_breast_cancer()


x = dataset.data


y = dataset.target

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)


scaler = StandardScaler()


x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
model = Sequential(layers=[
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')

    ])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=20 , epochs=100)

model.save('model.keras')
y_pred = model.predict(x_test)

y_pred = [int(x) for x in y_pred]


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
