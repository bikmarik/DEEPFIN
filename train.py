import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LOAD THE TRAINING DATA
X = np.load("data/X_train.npy")
Y = np.load("data/Y_train.npy") 
# 2. SCALING
scaler = StandardScaler()
X_flat = X.reshape(-1, X.shape[-1])
X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

Y_flat = Y.reshape(-1, Y.shape[-1])
Y_scaled = scaler.transform(Y_flat).reshape(Y.shape)

# 3. DEFINE THE ENCODER-DECODER (Seq2Seq)
n_features = X.shape[2] # 12 features from your C++ Unified Tensor

model = keras.Sequential([
    # ENCODER: Compresses 3 years of history into a "Financial Context" vector
    layers.Input(shape=(3, n_features)),
    layers.LSTM(128, activation='tanh', return_sequences=False),

    # BRIDGE: Repeats the context for each of the 3 years we want to predict
    layers.RepeatVector(3),
    
    # DECODER: Unpacks the context into 3 future years
    layers.LSTM(128, activation='tanh', return_sequences=True),
    
    # OUTPUT: Map the hidden states back to our 12 financial features
    layers.TimeDistributed(layers.Dense(n_features))
])

model.compile(optimizer="adamw", loss="mse", metrics=["mae"])
model.summary()

# 4. TRAIN
print("Training DEEPFIN Intelligence...")
history = model.fit(
    X_scaled, Y_scaled, 
    epochs=100, 
    batch_size=4, 
    validation_split=0.1,
    verbose=1
)

# 5. SAVE FOR SIMULATION
model.save("weights/deepfin_v2.keras")
np.save("data/scaler_mean.npy", scaler.mean_)
np.save("data/scaler_scale.npy", scaler.scale_)

# 6. VISUALIZE TRAINING LOSS
plt.plot(history.history['loss'], label='Loss')
plt.title('DEEPFIN Model Convergence')
plt.legend()
plt.savefig('assets/training_loss.png')