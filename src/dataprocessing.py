from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(test_size=0.2, val_size=0.2, random_state=0):
    '''
    Return the train set for training.
    '''
    X, y = load_diabetes(return_X_y=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_val, _, y_train_val, _ = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, y_train, y_val

def get_test_data(test_size=0.2, random_state=0):
    '''
    Return the test set for scoring.
    '''
    X, y = load_diabetes(return_X_y=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_test, y_test