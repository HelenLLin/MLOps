import pandas as pd
import numpy as np 
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split

def load_data(data_name):
    # Load data
    data = pd.read_csv(data_name)
    return data

def process_data(train_data, test_data, chi2percentile):
    # Strip whitespace from column names
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()

    # Drop where target (life expectancy) is na
    train_data = train_data.dropna(subset=['Life expectancy'])
    test_data = test_data.dropna(subset=['Life expectancy'])

    # Get target variables
    train_y, test_y = train_data['Life expectancy'], test_data['Life expectancy']

    # Drop identifying columns and target
    train_data = train_data.drop(columns=['Country', 'Life expectancy'])
    test_data = test_data.drop(columns=['Country', 'Life expectancy'])

    # Create pipeline for imputing and scaling numeric variables
    # one-hot encoding categorical variables, and select features based on chi-squared value
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(f_regression, percentile=chi2percentile)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include = ['int', 'float'])),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude = ['int', 'float'])),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor)]
    )

    # Create new train and test data using the pipeline
    clf.fit(train_data, train_y)
    train_new = clf.transform(train_data)
    test_new = clf.transform(test_data)

    # Transform to dataframe and save as a csv
    train_new = pd.DataFrame(train_new)
    test_new = pd.DataFrame(test_new)
    train_new['y'] = train_y
    test_new['y'] = test_y
    return train_new, test_new, clf

def save_data(train_new, test_new, train_name, test_name, clf, clf_name):
    train_new.to_csv(train_name)
    test_new.to_csv(test_name)
    
    # Save pipeline
    with open(clf_name,'wb') as f:
        pickle.dump(clf,f)

if __name__=="__main__":
    
    params = yaml.safe_load(open("params.yaml"))["features"]

    data_name = params["data_path"]
    chi2percentile = params["chi2percentile"]
    print('--- Params Loaded ---')

    data = load_data(data_name)
    print('--- Data Loaded ---')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_new, test_new, clf = process_data(train_data, test_data, chi2percentile)
    print('--- Data Processed ---')
    save_data(train_new, test_new, 'data/processed_train_data.csv', 'data/processed_test_data.csv', clf, 'data/pipeline.pkl')
    print('--- Data Saved ---')