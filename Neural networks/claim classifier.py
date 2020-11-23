################################################################################
# Author: Maleakhi, Faidon, Harry, Jamie
# Date: 18/November/2019
# Description: A claim classifier to predict the probability of a contract 
# making a claim. We will predict "made_claim" attribute which is a binary
# classification task. 
################################################################################
import numpy as np
import pandas as pd
import pickle

# Keras Library
from keras.models import Model
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Scikit Learn Library
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors

# Matplotlib
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
## ClaimClassifier Model

class ClaimClassifier:
    """
    Claim Classifier model used to predict made_claim. 
    """
    
    #---------------------------------------------------------------------------
    ## Class Main Methods

    def __init__(self, params=None):
        """
        Constructor of ClaimClassifier (initialise Keras model).

        Parameters
            model -- Keras model
        """
        if not params:
            # Do not have model, in some situation useful to access _preprocessor
            # functionality only
            self.model = None
        else:
            # Initialise keras model
            self.model = KerasClassifier(self.create_model, **params)

        # Used for data normalisation
        self.min_max_scaler = None

        # Store useful values for fit and predict
        self.params = params # store tuning result
        self.optimal_threshold = None 
        self.y_predict_proba = None # used to plot roc curve for evaluation

    def _preprocessor(self, X_raw):
        """
        Data preprocessing function.
        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
            X_raw {np.ndarray} -- data to be normalised

        Returns
        -------
            {np.ndarray} -- normalised version of the data
        """

        # Define the transformation based on entire dataset first
        if not self.min_max_scaler:
            self.min_max_scaler = preprocessing.MinMaxScaler()
            X_normalised = self.min_max_scaler.fit_transform(X_raw)
        
        # Perform normalisation
        else:
            X_normalised = self.min_max_scaler.transform(X_raw)

        return  X_normalised

    def fit(self, X_raw, y_raw):
        """
        Classifier training function.
        Training function for your classifier.

        Parameters
            X_raw {np.ndarray} -- raw data as downloaded
            y_raw {np.ndarray} -- one dimensional numpy array (binary target variable)
        """
        # Normalise the data first
        X_raw = self._preprocessor(X_raw)

        # Upsample data using smote
        X_train_smoted, y_train_smoted = self.upsample(X_raw, y_raw)

        # Train model
        self.model.fit(X_train_smoted, y_train_smoted, epochs=self.params["epochs"], batch_size=32, validation_split=0.1)

        # Get best threshold (for testing based on training data
        y_predict_smoted = self.model.predict_proba(X_train_smoted)[:,1]
        self.optimal_threshold = self.get_optimal_threshold(y_predict_smoted, y_train_smoted)

    def predict(self, X_raw):
        """
        Classifier probability prediction function.

        Implement the predict function for classifier.

        Parameters
            X_raw {np.ndarray} -- raw data as downloaded

        Returns
            {np.ndarray} -- A one dimensional array of 0 and 1
        """
        # Prediction with model
        X_test_normalised = self._preprocessor(X_raw)

        # Return predict probability
        y_predict_proba = self.model.predict_proba(X_test_normalised)[:,1]
        self.y_predict_proba = y_predict_proba # used to plot roc curve
        y_predict = np.array([1 if y >= self.optimal_threshold else 0 for y in y_predict_proba])

        return y_predict

    def evaluate_architecture(self, y_predict, y_test):
        """
        Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        Parameters
            y_predict {np.ndarray} -- 0 or 1 array (prediction)
            y_test -- 0 or 1 array (ground truth)
        """
        # Plot confusion matrix
        labels = [0, 1] # binary classification
        try:    
            self.plot_roc(self.y_predict_proba, y_test) # plot roc curve and save it in a roc.png file
        except Exception:
            pass # do nothing

        # Print confusion_matrix, and classification_report
        print("\nConfusion Matrix------------------------------------------------\n")
        cm = confusion_matrix(y_test, y_predict, labels=labels)
        print(cm)
        print("\nConfusion Matrix Normalised-------------------------------------\n")
        cm_normalised = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print(cm_normalised)
        print("\nClassification Report-------------------------------------------\n")
        print(classification_report(y_test, y_predict, labels))
    
    #---------------------------------------------------------------------------
    ## Class Helper Methods

    def create_model(self, hidden_layers, neurons, input_dim):
        """
        (Sklearn) Wrapper keras model so that we can perform scikit functions 
        on the model.

        Parameters
            hidden_layers {int} -- number of hidden layers
            neurons {int} -- number of neurons in every layer
            input_dim {int} -- input dimension of the data

        Returns
            model -- Scikit keras wrapper model
        """
        # Set up keras neural network
        inputs = Input(shape=(input_dim,))
        x = inputs

        # Setup model
        # By default, always use relu activation functions
        # Homogenise number of neurons for every layer
        for _ in range(hidden_layers):
            x = Dense(neurons, activation="relu")(x)

        # Output layer
        x = Dense(1, activation="sigmoid")(x)
        
        # Generate model
        model = Model(inputs, x)
        
        # Compile model
        model.compile(
            optimizer="adam",
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
        return model
    
    def upsample(self, X_train, y_train):
        """
        Upsamples algorithm based on SMOTE
        Takes as input the data set to train on.
        1) Calculates the nearest neighbours of a random point in the minority class.
        2) Calculates the vector which joins the two points.
        3) Adds a random fraction of this vector to the random point
        4) This is the new data point.

        Parameters
            X_train {np.ndarray} -- training input
            y_train {np.ndarray} -- training labels

        Returns
            {np.ndarray} -- smoted training input (upsampled)
            {np.ndarray} -- smoted training labels (upsampled)
        """
        # First convert to pandas to make sure we can process the data accordingly
        # If not pandas, convert to pandas (if already pandas proceed)
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
    
        # Get the indices of the minority class
        minority_sample_indices = y_train.index[y_train.iloc[:] == 1].tolist()
        minority_sample = X_train.loc[minority_sample_indices,:]

        # Calculate the nearest neighbour of the minority class
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(minority_sample)
    
        # Number of additional values needed:
        num_updates = (y_train == 0).astype(int).sum(axis=0) - (y_train == 1).astype(int).sum(axis=0)

        new_data_x = []
        new_data_y = []
        # Until we have a balanced data set (/4 as create 4 points between each nearest neighbour)
        for _ in range(int(num_updates/4)):
            # Choose a random data point form the minority class
            random_indices =  np.random.choice(minority_sample_indices)
            random_point = X_train.loc[random_indices,:]

            # Calculate its nearest neighbors
            nearest_neigh = neigh.kneighbors([list(random_point.to_numpy())], return_distance=False)
            nearest_neigh = (nearest_neigh[0])[0]
            nearest_neigh = minority_sample.index[nearest_neigh]

            # Calculate the vector between each points
            vec_between_points = X_train.loc[random_indices,:] - X_train.loc[nearest_neigh,:]
            # Generate 4 random points along the vector
            for _ in range(4):
                # First need to generate a random number between 0 and 1 do deside where the new point will lie on the vector.
                random_point = np.random.uniform(0, 1)
                random_position = random_point * vec_between_points
                new_data_point = X_train.loc[nearest_neigh,:] + random_position
                new_data_point = list(new_data_point.to_numpy())
                new_data_x.append(new_data_point)
                new_data_y.append([1])

        # Convert back to a dataframe
        df_new_data_x = pd.DataFrame(new_data_x, columns = list(X_train.columns))
        df_new_data_y = pd.DataFrame(new_data_y)
    
        # Add the new data to the X and Y training data
        X_train = X_train.append(df_new_data_x)
        y_train = y_train.append(df_new_data_y)

        # Return the ndarray version
        return np.array(X_train), np.array(y_train)
    
    def plot_roc(self, y_predict, y_test):
        """
        Plot ROC (Receiver Operating Characteristic).
        
        Parameters
            y_predict {np.ndarray} -- array of probabilities
            y_test {np.ndarray} -- array of true labels
        """
        # Calculate fpr and tpr for all threshold of the classification
        fpr, tpr, thresholds = roc_curve(y_test, y_predict)
        roc_auc = auc(fpr, tpr)

        # Plot using matplotlib
        plt.title("Receiver Operating Characteristic")
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("roc.png")
        plt.close()
    
    def get_optimal_threshold(self, y_predict, y_label):
        """
        ROC (Receiver Operating Characteristic) based threshold finder.
        We will used this to get optimal threshold to create a hard classification
        replacing the soft probability result.

        Parameters
            y_predict {np.ndarray} -- array of probabilities
            y_label {np.ndarray} -- array of true label
        
        Returns
            optimal_threshold {float} -- optimal threshold value from roc curve
        """
        # Calculate fpr and tpr for all threshold of the classification
        fpr, tpr, thresholds = roc_curve(y_label, y_predict)
        optimal_index = np.argmax(tpr-fpr)
        optimal_threshold = thresholds[optimal_index]

        # Return the best threshold according to the plot
        return optimal_threshold
    
    def save_model(self):
        """
        Save the model to part2_claim_classifier.pickle
        """
        with open("part2_claim_classifier.pickle", "wb") as target:
            pickle.dump(self, target)

#-------------------------------------------------------------------------------
## Hyper-parameter Tuning

def ClaimClassifierHyperParameterSearch(X_train, y_train):
    """
    Performs a hyper-parameter for fine-tuning the classifier. We will tune
    number of epochs, neurons, layers.

    Parameters
        X_train {np.ndarray} -- input dataset
        y_train {np.ndarray} -- input label

    Returns
        {dict} -- Best parameters dictionary
    """
    # Normalise the data first
    classifier = ClaimClassifier() # dummy classifier, used for _preprocessor
    X_train = classifier._preprocessor(X_train)

    # Upsample the data
    X_train_smoted, y_train_smoted = classifier.upsample(X_train, y_train)

    # Model wrapper
    classifier = KerasClassifier(classifier.create_model, verbose=0) # silent message
    
    # Parameter to be search
    # Feel free to add few stuffs inside the list in the dictionary to grid search
    # multiple values.
    param_grid = {
        "epochs": [10],
        "input_dim": [9],
        "hidden_layers": [5],
        "neurons": [100],
    }

    # Perform grid search
    print("Begin grid search---------------------------------------------------")
    ss = StratifiedKFold(n_splits=3, shuffle=True)
    grid_search = GridSearchCV(classifier, param_grid=param_grid, scoring='roc_auc', cv=ss)
    grid_result = grid_search.fit(X_train_smoted, y_train_smoted)

    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print("-------------------------------------------------------------------")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result.best_params_
    
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    The following is step by step on how to train the deep learning model.
    """
    # Import raw data
    raw_data = pd.read_csv('part2_data.csv')
    
    # Get data (not including claim amount)
    X_dataset = raw_data.iloc[:,0:9]
    y_dataset = raw_data.iloc[:,10]
    
    # Split the dataset in training and test
    X_train, X_test, y_train, y_test = train_test_split(X_dataset,y_dataset,test_size = 0.1)
    # Convert to nparray to replicate test scenario
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Grid search tuning
    best_params = ClaimClassifierHyperParameterSearch(X_train, y_train)
    # If you do not want to do grid search tuning, just comment out previous line
    # and use this instead.
    # best_params = {
        # "epochs": 10,
        # "input_dim": 9,
        # "hidden_layers": 5,
        # "neurons": 100,
    # }
    classifier = ClaimClassifier(best_params)

    # Pass training dataset
    classifier.fit(X_train, y_train)
    
    # Predict test dataset
    y_predict = classifier.predict(X_test)
    
    # Evaluate the model on test data (roc plot, confusion matrix, normalised cm, classification report)
    classifier.evaluate_architecture(y_predict, y_test)
    
    # Save model
    # classifier.save_model()
