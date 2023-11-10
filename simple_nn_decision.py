'''
A script to build and visualize a neural network using Tensorflow, Keras and SciKit-Learn
Also available as a Jupyter Notebook:
https://github.com/jacobtohahn/simple-nn-viz/blob/main/simple_nn_decision.ipynb
'''

import math, random, os, time
from numpy import array, vstack, linspace
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import DecisionBoundaryDisplay
from scikeras.wrappers import KerasClassifier
from tensorflow import get_logger, autograph, device
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from absl import logging
from tqdm.auto import tqdm

# Set all logging to error level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
get_logger().setLevel('ERROR')
autograph.set_verbosity(0)
logging.set_verbosity(logging.ERROR)

# Set the number of samples and noise
samples, noise = 5000, 0.3

# Visualize the decision boundary per epoch. Slow.
show_progress = True

def classification_data(samples=2000):
    X, y = make_classification(samples, 2, n_informative=2, n_redundant=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X, y, X_train, X_test, y_train, y_test

def spiral_data(samples=2000, noise=0.3):
    points = []
    n = samples/2
    def randUniform(a, b):
        return random.random() * (b - a) + a
    def genSpiral(deltaT, label):
        i = 0
        while i < n:
            r = i / n * 5
            t = 1.75 * i / n * 2 * math.pi + deltaT
            x = r * math.sin(t) + randUniform(-1, 1) * noise
            y = r * math.cos(t) + randUniform(-1, 1) * noise
            points.append([x, y, label])
            i = i + 1
    genSpiral(0, 1)
    genSpiral(math.pi, 0)
    X = array(points)[:,:2]
    y = array(points)[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X, y, X_train, X_test, y_train, y_test

X, y, X_train, X_test, y_train, y_test = spiral_data(samples) # <--- Call either classification_data(samples) or spiral_data(samples, noise)

# Create a colormap for the output colors
newcolors = vstack((linspace((242/255, 151/255, 31/255), (1, 1, 1)), linspace((1, 1, 1), (68/255, 145/255, 227/255))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')

# Function for creating the model called by KerasClassifier()
def create_model(learning_rate=0.01):
    # Define the Keras model
    model = Sequential()

    # Add some layers and nodes
    model.add(Dense(8, input_shape=(2,), activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    # Compile the Keras model
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model

# Create a callback to visualize progress
class ShowDisplayBound(Callback):
    # Runs at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        if epoch%5 == 0:
            # Get the model's prediction using the test data
            predicted_y = pipeline.predict(X_test)
            difference = differences(y_test, predicted_y)
            accuracy = (samples/4 - difference) / (samples/4) * 100
            # If show_progress is True, visualize the model's decision boundary for each epoch
            # The decision boundary shows the model's prediction probability for a 100x100 grid over the domain of the dataset
            if show_progress == True:
                # Clear the previous epoch's plot
                ax.clear()
                # Plot the decision boundary of the model
                DecisionBoundaryDisplay.from_estimator(pipeline, X_test, grid_resolution=100, cmap=newcmp, alpha=1, ax=ax)
                # Plot the predicted data points
                ax.scatter(X_test[:,0], X_test[:,1], c=predicted_y, cmap=newcmp, marker='o', edgecolors="black")
                ax.set_title("Predicted Output:")
                fig.suptitle(f"Epoch {epoch}\nAccuracy: {round(accuracy, 2)}%")
                # Show the plot
                plt.gcf()
                plt.pause(0.001)
            # Print progress bar and accuracy
            print(f"{tqdm.format_meter(epoch, epochnum, time.time() - t_start)} | Accuracy: {round(accuracy, 2)}%", end="\r", flush=True)
        if epoch == epochnum - 1:
            predicted_y = pipeline.predict(X_test)
            difference = differences(y_test, predicted_y)
            accuracy = (samples/4 - difference) / (samples/4) * 100
            ax.clear()
            DecisionBoundaryDisplay.from_estimator(pipeline, X_test, grid_resolution=100, cmap=newcmp, alpha=1, ax=ax)
            ax.scatter(X_test[:,0], X_test[:,1], c=predicted_y, cmap=newcmp, marker='o', edgecolors="black")
            ax.set_title("Predicted Output:")
            fig.suptitle(f"Epoch {epoch}\nAccuracy: {round(accuracy, 2)}%")
            plt.gcf()
            plt.pause(0.001)
            
# Set the number of training epochs
epochnum = 750

# Create the KerasClassifier model. -1 batch size means all samples per epoch.
clf = KerasClassifier(model=create_model, batch_size=-1, epochs=epochnum, learning_rate=0.02, verbose=0, callbacks=[ShowDisplayBound])

# Create a list of estimators
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('clf', clf))

# Create a Pipeline with the estimators
pipeline = Pipeline(estimators)

# Helper function for calculating accuracy
def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))

# Set up the plots for visualization
global fig, ax
fig = plt.figure(figsize=(11,5))
# One subplot for the prediction and one for the dataset
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(X[:,0], X[:,1], c=y, cmap=newcmp, marker='o', edgecolors="black")
ax2.set_title("Original dataset:")

# Fit the model on the training data. Each epoch, the callback ShowDisplayBound() is executed.
with device("CPU"):
    t_start = time.time()
    pipeline.fit(X_train, y_train)

plt.show()