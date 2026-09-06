from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example input dimensions
timesteps = 10
features = 5

# Build preliminary LSTM model
model = Sequential([
    LSTM(64, input_shape=(timesteps, features)),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
