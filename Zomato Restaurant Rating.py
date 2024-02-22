# Importing Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Loading Dataset
data=pd.read_csv("zomato_data.csv")

# Splitting the data
x=data.drop('rate',axis=1)
y=data['rate']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)

# Loading Model
model=pickle.load(open('model.pkl','rb'))

# Training the model
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

# Prediction
print(y_pred)
