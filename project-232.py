from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense

#Loading the dataset

dataset=loadtxt('pokemon.csv',delimiter=',')

x=dataset[:,1:7]

y=dataset[:,8]

print("Value of input layer : ",x)

print("Value of output layer : ",y)

model=Sequential()

model.add(Dense(12,input_dim=6,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x,y,epochs=250,batch_size=100)

predictions=model.predict(x)

for i in range(785,800):
	print(f'{x[i].tolist()} => {predictions[i]} expected {y[i]}')