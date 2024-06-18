import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler


# Load all columns from the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


# Load only specific columns from the dataset
def load_data_includeCols(filepath, includeCols):
    data = load_data(filepath)
    refined_data = data[includeCols]
    return refined_data


# Load all columns from the dataset EXCEPT specific excluded columns
def load_data_excludeCols(filepath, excludeCols):
    data = load_data(filepath)
    includeCols = [col for col in data.columns if col not in excludeCols]
    refined_data = data[includeCols]
    return refined_data


# Handle missing values in a DataFrame using imputation
def meanImpute(df):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Define features and target
def defineXY(df, target):
    y = df[target]
    X = df.drop(target, axis=1)
    return X, y


def performPCA(X, n_components):
    """
    Perform PCA on the dataset and return the transformed features.

    Parameters:
    X (pd.DataFrame): The input features.
    n_components (int): Number of principal components to keep.

    Returns:
    pd.DataFrame: The transformed dataset with reduced dimensions.
    PCA: The fitted PCA object.
    """

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Performing PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Creating a DataFrame with the principal components
    pc_df = pd.DataFrame(data=principal_components,
                         columns=[f'PC{i+1}' for i in range(n_components)])

    # Explained variance by each principal component
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance by each principal component:")
    for i, variance in enumerate(explained_variance):
        print(f"PC{i+1}: {variance:.4f}")

    print(pc_df.head())
    return pc_df, pca


def performPCA_separateSets(X_train, X_test, n_components, seed):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca


# Split data into train and test sets
def splitTrainingTesting(X, y, test_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=seed)
    return X_train, X_test, y_train, y_test


def createRF(X_train, X_test, y_train, y_test, params, seed):
    model = RandomForestRegressor(random_state=seed, **params)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

    results = y_pred - y_test
    unique, counts = np.unique(results, return_counts=True)

    # Plot residuals
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(results, bins=50)
    ax.set(xlabel='Error', ylabel='Counts',
           title='Distribution of Prediction Errors')
    ax.grid(True)
    plt.show()
    return model


# Custom scorer function to calculate adjusted R^2
def adjusted_r2_scorer(estimator, X, y):
    print(X)
    print(y)
    print(estimator.predict(X))
    r2 = r2_score(y, estimator.predict(X))
    n = X.shape[0]
    p = X.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2


def hypertune_model(X_train_pca, y_train):
    # Make a adj-r^2 scorer from the custom scorer function:
    # adjusted_r2 = make_scorer(adjusted_r2_scorer, greater_is_better=True)

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 500, 1000, 1500, 2000],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'max_leaf_nodes': [None, 2, 5, 10, 15, 20]
    }

    grid_search = GridSearchCV(model, param_grid, verbose=3,
                               cv=LeaveOneOut(), n_jobs=-1,
                               scoring='neg_mean_squared_error',)
                            
    grid_search.fit(X_train_pca, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best neg. mean squared error (MSE): {grid_search.best_score_:.4f}")

    return grid_search.best_params_


# show all stats for features used in model (importance, correlation, ranking)
def allFeatureStats(model, data, X, X_train, y_train, target):
    featureImportance(model, X)
    correlationMatrix(data, target)
    rfe(model, X_train, y_train, X)
    anova(X_train, y_train, X)


# Feature Importances
def featureImportance(model, X):
    print("\nImportance of each feature used in the model:")
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    for idx in sorted_indices:
        print(f"{X.columns[idx]}: {feature_importances[idx]}")


# Correlation matrix of the dataframe
def correlationMatrix(data, target):
    print("\n\nCorrelation matrix of the dataframe, showing the Pearson" +
          "correlation coefficients between each pair of features:")
    correlation_matrix = data.corr()
    target_correlation = correlation_matrix[target].abs().sort_values(
        ascending=False
        )
    print(target_correlation)


# Features ranked via RFE
def rfe(model, X_train, y_train, X):
    rfe = RFE(model, n_features_to_select=1)
    rfe.fit(X_train, y_train)
    ranked_features = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_),
                                 X.columns))
    print("\n\nFeatures ranked via RFE:")
    print(ranked_features)


# Features ranked via ANOVA F-value
def anova(X_train, y_train, X):
    from sklearn.feature_selection import SelectKBest, f_regression
    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X_train, y_train)
    ranked_features = sorted(zip(map(lambda x: round(x, 4), selector.scores_),
                                 X.columns))
    print("\n\nFeatures ranked via ANOVA F-value:")
    print(ranked_features)
