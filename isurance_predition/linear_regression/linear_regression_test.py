import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from linear_regression_fc import LinearRegressionFC


def data_processing(df):
    ohe = OneHotEncoder()
    ohe_feature = ohe.fit_transform(df[['region']]).toarray()
    feature_name = ohe.categories_
    ohe_df = pd.DataFrame(ohe_feature, columns=feature_name)
    df = pd.concat([df, ohe_df], axis=1)
    df['sex'] = np.where(df['sex'] == 'male', 1, 0)
    df['smoker'] = np.where(df['smoker'] == 'yes', 1, 0)
    df.drop(['region'], axis=1, inplace=True)

    charges_col = df.pop('charges')
    df.insert(df.shape[1], 'charges', charges_col)

    normalize_dataset = StandardScaler().fit_transform(df)
    df = pd.DataFrame(normalize_dataset)

    return df


if __name__ == '__main__':
    df = pd.read_csv(r'D:\in\ml\gp\ml\final\insurance.csv')

    # Linear Regression from scratch test
    dataset = data_processing(df).values.tolist()
    y = []
    for row in dataset:
        y.append(row[-1])
        row.pop()
    X = dataset

    X_train, X_val = X[:1001], X[1001:]
    y_train, y_val = y[:1001], y[1001:]
    lrfc = LinearRegressionFC()
    print('--- Linear Regression from scratch test ---')
    lrfc.fit(X, y, 0.02, 200, X_val, y_val)
    print(f'RMSE:', lrfc.evaluate(X_val, y_val))

    # Compare with Linear Regression from sklearn
    from sklearn import linear_model
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    print('\n--- Linear Regression from sklearn test ---')

    dataset2 = data_processing(df).to_numpy()

    X_train, X_eval, y_train, y_eval = dataset2[:1001, :-1], dataset2[1001:, :-1], dataset2[:1001, -1], dataset2[1001:, -1]
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_eval_predict = lr.predict(X_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, y_eval_predict))
    print('RMSE:', rmse)

    y_val_predict = lr.predict(X_eval)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(lrfc.train_loss, color='b')
    axis[0, 0].plot(lrfc.val_loss, color='r')
    axis[0, 0].set_xlabel('epochs')
    axis[0, 0].set_ylabel('loss')
    axis[0, 0].legend(['train_loss', 'eval_loss'], loc='upper right')
    axis[0, 0].set_title("Loss")

    axis[0, 1].scatter(y_val, lrfc.y_predict, color='r', alpha=0.5)
    axis[0, 1].plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='b')
    axis[0, 1].set_xlabel('y_val')
    axis[0, 1].set_ylabel('y_predict')
    axis[0, 1].set_title("Y valuate vs Prediction with mllearn")

    axis[1, 0].scatter(y_eval, y_val_predict, color='r', alpha=0.5)
    axis[1, 0].plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='b')
    axis[1, 0].set_xlabel('y_val')
    axis[1, 0].set_ylabel('y_predict')
    axis[1, 0].set_title("Y valuate vs Prediction with sklearn")

    plt.show()
