import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

data=pd.read_csv("E:\Data Science\ML Projects\Zomato Restaurant Rating\zomato_data.csv")

x=data.drop('rate',axis=1)
y=data['rate']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)

model=pickle.load(open('E:\Data Science\ML Projects\Zomato Restaurant Rating\model.pkl','rb'))

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(y_pred)