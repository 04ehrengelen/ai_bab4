#Example 4.4 Multiple Layer Perceptron Classification
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
test_size = 0.3, random_state = 7)
features = X_train.shape[1]
#MPL 3-layer Model ==============================================
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=[X_train
.shape[1]]))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
print(model.summary())
print(model.get_config())
model.compile(optimizer = 'adam', loss = 'mean_squared_error',
metrics = ['mse'])
model.fit(X_train, y_train, batch_size = 50, validation_split=0.2,
epochs = 100, verbose = 1)
results = model.evaluate(X_test, y_test)
print('loss: ', results[0])
print('accuracy: ', results[1])