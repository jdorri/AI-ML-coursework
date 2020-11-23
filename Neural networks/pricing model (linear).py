################################################################################
# Author: Maleakhi, Faidon, Harry, Jamie
# Date: 18/November/2019
# Description: Pricing model which is used for insurance pricing. This model will
# be used in a competitive AI market. (Logistic Regression)
################################################################################
import numpy as np
import pandas as pd 
import pickle

# Scikit Learn Library
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

#-------------------------------------------------------------------------------
## Calibrator Function

def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier

#-------------------------------------------------------------------------------
## Pricing Model Class
class PricingModelLinear():

    #---------------------------------------------------------------------------
    ## Class Main Methods

    def __init__(self, calibrate_probabilities=False):
        """
        Constructor for the pricing model class.
        """
        self.y_median = None # Median used to compute mean
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================

        # Used default parameter for logistic regression        
        self.base_classifier = LogisticRegression()

        # Used for data normalisation
        self.min_max_scaler = None

        # Variables to set standard columns for preprocessing
        self.columns_set = False
        self.column_names = None
        self.column_all_names = None
        self.column_all_dtypes = None

    def _preprocessor(self, X_raw):
        """
        Data preprocessing function.
        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
            X_raw {np.ndarray} -- An array, this is the raw data as downloaded

        Returns
            {np.ndarray} -- A clean data set that is used for training and prediction.
        """
        # Step 1: Convert to pandas for manipulation
        X_data = pd.DataFrame(X_raw, columns=self.column_all_names).astype(self.column_all_dtypes)
        
        # Step 2: Drop columns that are not needed
        # We drop this using domain knowledge
        X_data = X_data.drop(columns=['id_policy','vh_make', 'vh_model', 'regional_department_code', 'pol_insee_code', 'commune_code', 'canton_code', 'city_district_code'])
        
        # Step 3: Encode classification data using OneHotEncoding
        if not self.columns_set:
            X_data_encoded = pd.get_dummies(X_data, prefix_sep='_', drop_first=False)
            self.column_names = X_data_encoded.columns 
            self.columns_set = True
        else:
            X_data_encoded = pd.get_dummies(X_data, prefix_sep='_', drop_first=False)
            # Ensure DataFrame has the same columns and same order as original data
            for column in self.column_names:
                if column not in X_data_encoded.columns: 
                    X_data_encoded[column] = 0
            X_data_encoded = X_data_encoded.loc[:, self.column_names]
        
        # Replace NaN values
        X_data_encoded = X_data_encoded.fillna(0)        
        
        # Step 4: Normalise all input data - define the transformation based on entire dataset first
        if not self.min_max_scaler:
            self.min_max_scaler = preprocessing.MinMaxScaler()
            X_normalised = self.min_max_scaler.fit_transform(X_data_encoded)
        else:
            X_normalised = self.min_max_scaler.transform(X_data_encoded)

        # Return numpy array
        return  np.array(X_normalised)

    def fit(self, X_raw, y_raw, claims_raw):
        """
        Classifier training function.

        Parameters
            X_raw {np.ndarray} -- This is the raw data as downloaded (training)
            y_raw {np.ndarray} -- A one dimensional array, this is the binary target variable
            claims_raw {np.ndarray} -- A one dimensional array which records the severity of claims

        Returns
            model -- an instance of the fitted model, but the model is already an instance variable,
                     so it is optional to take the model here.
        """
        # Get the mean of the claim_raw data (used for severity model)
        non_zero_claims_indices = np.where(claims_raw != 0)[0]
        self.y_median = np.median(claims_raw[non_zero_claims_indices])

        # Pre-process data
        X_clean = self._preprocessor(X_raw)

        # Upsample data using SMOTE based method
        X_train_smoted, y_train_smoted = self.upsample(X_clean, y_raw)

        # Train base_classifier
        self.base_classifier.fit(X_train_smoted, y_train_smoted.ravel())

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        # if self.calibrate:
        #     self.base_classifier = fit_and_calibrate_classifier(
        #         self.base_classifier, X_clean, y_raw)
        # else:
        #     self.base_classifier = self.base_classifier.fit(X_clean, y_raw)

        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """
        Classifier probability prediction function.

        Parameters
            X_raw {np.ndarray} -- This is the raw data as downloaded

        Returns
            {np.ndarray} -- A one dimensional array of probability getting POSITIVE class
        """
        # Preprocess data
        X_test_normalised = self._preprocessor(X_raw)

        # Return predict probability
        y_predict = self.base_classifier.predict_proba(X_test_normalised)[:, 1] # to get the probability predicting 1        

        return np.array(y_predict) # return probabilities for the positive class (label 1)


    def predict_premium(self, X_raw):
        """
        Predicts premiums based on the pricing model.

        Parameters
            X_raw {numpy.ndarray} -- A numpy array, this is the raw data as downloaded

        Returns
            numpy.ndarray -- A one dimensional array of probability getting POSITIVE class
        """
        # Add premium of 1%
        return np.array(self.predict_claim_probability(X_raw) * self.y_median * 1.01)

    #---------------------------------------------------------------------------
    ## Class Helper Methods
    
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
    
    def save_model(self):
        """Saves the class instance as a pickle file."""
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    The following is step by step on how to train the deep learning model.
    """
    # Initialise classifier
    pricing_model = PricingModelLinear()

    ## Data Preprocessing using pandas
    # Load data to Pandas
    raw_data = pd.read_csv('part3_data.csv')
    # Input data is everything apart from target columns
    X_raw = raw_data.drop(columns=['made_claim', 'claim_amount'])
    # y data is the binary classification
    y_raw = raw_data.loc[:,'made_claim']
    # Claim amount
    claims_raw = raw_data.loc[:, 'claim_amount']
    # Hold all column names
    pricing_model.column_all_names = X_raw.columns
    pricing_model.column_all_dtypes = X_raw.dtypes

    # Split the dataset in training and test
    X_train, X_test, y_train, y_test = train_test_split(X_raw,y_raw, test_size = 0.1)
    # Convert all to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    claims_raw = np.array(claims_raw)

    # Training our model
    pricing_model.fit(X_train, y_train, claims_raw)

    # Predict test dataset
    # If you wished to check the prediction result, uncomment these 2 lines
    y_predict = pricing_model.predict_claim_probability(X_test)
    y_predict_premium = pricing_model.predict_premium(X_test)
    print(f"The predicted probability using logistic regression is {y_predict}.")
    print(f"The predicted premium using logistic regression is {y_predict_premium}.")

    # Store model to pickle file:
    # pricing_model.save_model()

