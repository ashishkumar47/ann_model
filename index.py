from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import pandas as pd
seed=8
np.random.seed(seed)
#import data
df=pd.read_csv("BBCN.csv")
array=df.values
#spliting the data
x=array[:,0:11]
y=array[:,11]
# build a model
model = Sequential()
model.add(Dense(15, input_dim=11, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#fit the model
model.fit(x, y, nb_epoch=200, batch_size=10)
# score the model
loss,acurracy= model.evaluate(x,y)
print(loss,acurracy*100)
model.save("my_model.h5")
new_model=load_model("my_model.h5")
predictions=new_model.predict([x])
print(np.argmax(predictions[0]))
#print(predictions)