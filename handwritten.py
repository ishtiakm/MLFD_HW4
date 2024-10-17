import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the ZipDigits.train file
file_path = 'ZipDigits.train'

# Load the data from the file
data = []

# Read the file line by line
with open(file_path, 'r') as file:
    for line in file:
        values = list(map(float, line.split()))  # Convert each line to a list of floats
        data.append(values)

# Convert the list to a NumPy array for easier manipulation
data = np.array(data)

# Split the data into labels (first column) and features (remaining columns)
labels = data[:, 0]      # Labels are the first column
features = data[:, 1:]   # Features are the remaining columns

# Display the shape of features and labels
print(f'Features shape: {features.shape}')
print(f'Labels shape: {labels.shape}')

# Step 1: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(X_train.shape)

def filter_relevant_data(X_train, y_train, X_test, y_test,digit):
    """
    Filters the dataset to keep only the samples corresponding to the given digit

    :param X_train: Training data (numpy array of shape (n_samples_train, n_features))
    :param y_train: Training labels (numpy array of shape (n_samples_train,))
    :param X_test: Test data (numpy array of shape (n_samples_test, n_features))
    :param y_test: Test labels (numpy array of shape (n_samples_test,))
    :return: X_train_f, y_train_f, X_test_f, y_test_f: Filtered training and test sets
             containing only samples where the labels matches the given digit (1 or 5).
    """
    # Filter the training data for labels 1 and 5
    train_filter = np.isin(y_train, [digit])
    X_train_f = X_train[train_filter]
    y_train_f = y_train[train_filter]

    # Filter the test data for labels 1 and 5
    test_filter = np.isin(y_test, [digit])
    X_test_f = X_test[test_filter]
    y_test_f = y_test[test_filter]

    # Return the filtered data
    return X_train_f, y_train_f, X_test_f, y_test_f

X_train_f1,y_train_f1,X_test_f1,y_test_f1=filter_relevant_data(X_train,y_train,X_test,y_test,1)
X_train_f5,y_train_f5,X_test_f5,y_test_f5=filter_relevant_data(X_train,y_train,X_test,y_test,5)

# Function to calculate vertical symmetry
def vertical_symmetry(image):
    left_half = image[:, :8]  # Left half (first 8 columns)
    right_half = np.fliplr(image[:, 8:])  # Right half, flipped horizontally
    symmetry = np.mean(np.abs(left_half - right_half))  # Mean absolute difference
    return symmetry

# Function to calculate horizontal symmetry
def horizontal_symmetry(image):
    top_half = image[:8, :]  # Top half (first 8 rows)
    bottom_half = np.flipud(image[8:, :])  # Bottom half, flipped vertically
    symmetry = np.mean(np.abs(top_half - bottom_half))  # Mean absolute difference
    return symmetry

# Function to calculate the width (using a threshold to determine pixel activation)
def width(image, threshold=0.5):
    # Count non-zero pixels (greater than threshold) along the columns
    width = np.sum(np.max(image > threshold, axis=0))
    return width

def intensity(image):
    return np.mean(image)

# Main function to apply feature transform
def feature_transform(X,func1,func2):
    """
    Applies feature transformations to each image in X and returns a 2D numpy array
    with a single feature value per image.
    
    :param X: numpy array of shape (n_samples, 256), where each row is a flattened 16x16 image
    :return: numpy array of shape (n_samples,) containing two features value per image
    """
    # Initialize list to store the transformed features
    transformed_features = []

    # Loop over each image (flattened) in X
    for i in range(X.shape[0]):
        image = X[i].reshape(16, 16)  # Reshape the 256 feature vector to 16x16
        
        # Calculate the feature transform
        #vertical_symmetry = calculate_vertical_symmetry(image)
        #horizontal_symmetry = calculate_horizontal_symmetry(image)
        #width = calculate_width(image)
        
        # For simplicity, let's combine these features (you can choose how to combine them)
        # For this example, we'll just use the width as the feature, but you can customize this
        #feature_value = func(image)
        
        # Append the calculated feature to the list
        features=[func1(image),func2(image)]
        transformed_features.append(features)

    # Convert the list to a numpy array and return
    return np.array(transformed_features)

X_train_new_1=feature_transform(X_train_f1,horizontal_symmetry,vertical_symmetry)
X_train_new_5=feature_transform(X_train_f5,horizontal_symmetry,vertical_symmetry)

def scatter_plot(X_train1,X_train5):
    
    plt.plot(X_train1[:,0],X_train1[:,1],'bo')
    plt.plot(X_train5[:,0],X_train5[:,1],'rx')
    plt.xlabel("Horizontal Symmetry")
    plt.ylabel("Vertical Symmetry")
    plt.legend(['Digit 1','Digit 5'])
    plt.savefig("scatter.png")
    plt.show()
    
scatter_plot(X_train_new_1,X_train_new_5)