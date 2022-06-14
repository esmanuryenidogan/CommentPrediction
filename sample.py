import pandas as pd #Kütüphanelerimizi ekliyoruz
import numpy as np

data = pd.read_csv("data.txt", engine='python') #Data setimizi okuyoruz

X = data.iloc[:, 0].values #Veriyi X e alıyoruz
y = data.iloc[:, 1].values #Çıktıyı y e alıyoruz

from sklearn.model_selection import train_test_split #Test kütüphanesi ekliyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0) #Test için datanın %25 ini, kalan %75 i eğitim için ayırıyoruz

from sklearn import preprocessing 
labelCode = preprocessing.LabelEncoder()
X_train=labelCode.fit_transform(X_train) #Elimizdeki veri text olduğu için sayısal değerlere dönüştürüyoruz.(eğitim verisi)
print(X_train)

X_test=labelCode.fit_transform(X_test) #Elimizdeki veri text olduğu için sayısal değerlere dönüştürüyoruz.(test verisi)
print(X_test)

X_train = np.reshape(X_train, (-1, 1)) 
X_test = np.reshape(X_test, (-1, 1)) #Modelimizi eğitmeden önce test ve eğitim verilerimizi 2 boyutlu diziye dönüştürüyoruz

from sklearn.naive_bayes import GaussianNB #Modelimizi eğitmek için Gauss Naive Bayes sınıflandırıcısını kullanıyoruz
classifer = GaussianNB()
classifer.fit(X_train, y_train) 

y_pred = classifer.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test)) #Modelin başarı, doğruluk skorunu yazdırıyoruz

predicted= classifer.predict([[0]]) # 0:ürün anlatıldığı gibi değil beğenmedim - Bir test girdisi sağlayarak çıktıyı tahmin ediyoruz.
print(predicted)
