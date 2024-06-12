import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import gradio as gr

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the dataset
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# One-hot encode the labels
Y_train = to_categorical(y_train, num_classes=10)
Y_test = to_categorical(y_test, num_classes=10)

# Define the CNN model
def create_model(learning_rate=0.001, num_hidden_units=128):
    model = models.Sequential()
    model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_hidden_units, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Train the model and capture the weights of the first layer
def train_model(learning_rate, num_hidden_units, epochs):
    model = create_model(learning_rate, num_hidden_units)
    history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=0)
    
    # Extract the weights of the first layer
    first_layer_weights = model.layers[0].get_weights()[0]
    
    # Save the model weights
    model.save_weights('mnist_cnn.h5')
    
    return model, history.history, first_layer_weights

# Visualize weights of the first layer
def plot_weights(weights, epoch):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[-1]:
            ax.imshow(weights[:, :, 0, i], cmap='viridis')
            ax.axis('off')
    plt.suptitle(f'Weights of First Layer at Epoch {epoch}')
    plt.show()

# Plot training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Define Gradio interface
def visualize(learning_rate, num_hidden_units, epochs):
    model, history, weights = train_model(learning_rate, num_hidden_units, epochs)
    plot_history(history)
    for epoch in range(1, epochs+1):
        plot_weights(weights, epoch)
    return "Training complete. See visualizations above."

# Function to predict custom images
def predict_digit(image):
    model = create_model()
    model.load_weights('mnist_cnn.h5')
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]

# Gradio interface for training and visualizing the model
interface = gr.Interface(
    fn=visualize,
    inputs=[
        gr.Slider(minimum=0.0001, maximum=0.01, step=0.0001, value=0.001, label="Learning Rate"),
        gr.Slider(minimum=32, maximum=256, step=32, value=128, label="Number of Hidden Units"),
        gr.Slider(minimum=1, maximum=50, step=1, value=30, label="Epochs")
    ],
    outputs="text",
    live=True,
    description="Adjust hyperparameters to visualize the impact on training performance and first layer weights."
)

# Gradio interface for predicting custom digits
image_input = gr.Image(image_mode='L', source='upload', invert_colors=True)
label_output = gr.Label(num_top_classes=1)

predict_interface = gr.Interface(
    fn=predict_digit,
    inputs=image_input,
    outputs=label_output,
    description="Upload an image of a digit (28x28) to see the model's prediction."
)

# Combine both interfaces
demo = gr.TabbedInterface([interface, predict_interface], ["Train Model", "Predict Digit"])
demo.launch(share=True)
