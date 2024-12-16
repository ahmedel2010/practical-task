from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
dataset = datasets.load_breast_cancer()


x = dataset.data


y = dataset.target

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)



model = SVC(kernel='rbf',C=0.1,gamma=0.001)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
