import numpy 
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras import (
    layers, 
    initializers, 
    optimizers, 
    callbacks,
    activations,
    datasets,
    Model,
    utils
)
from catboost import CatBoostRegressor, Pool

SEED = 42
tf.random.set_seed(SEED)

(x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data(seed=SEED, test_split=0.3)

def split(array, ratio): return np.split(array, [int(len(array) * ratio)])
x_validation, x_test = split(x_test, 0.66)
y_validation, y_test = split(y_test, 0.66)

print('Train samples:', len(x_train))
print('Validation samples:', len(x_validation))
print('Test samples:', len(x_test))

'''
X struct

CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
LSTAT - % lower status of the population

Y struct

MEDV - Median value of owner-occupied homes in $1000's 
'''

LABELS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

def deep_neural_network():
    n_in = layers.Input(shape=(13,))
    n = layers.Dense(64, activation='elu', kernel_initializer=initializers.he_uniform())(n_in)
    n = layers.Dense(128, activation='elu', kernel_initializer=initializers.he_uniform())(n)
    n = layers.Dense(128, activation='elu', kernel_initializer=initializers.he_uniform())(n)
    n = layers.Dense(256, activation='elu', kernel_initializer=initializers.he_uniform())(n)
    n = layers.Dense(128, activation='elu', kernel_initializer=initializers.he_uniform())(n)
    n = layers.Dense(128, activation='elu', kernel_initializer=initializers.he_uniform())(n)
    n = layers.Dense(64, activation='elu', kernel_initializer=initializers.he_uniform())(n)
    n_out = layers.Dense(1, activation='linear')(n)

    model = Model(inputs=n_in, outputs=n_out)
    utils.plot_model(model, 'deep_neural_network.png', show_shapes=True)

    model = _fit_model(model, x_train, y_train, x_validation, y_validation)

    return model

def _fit_model(model, x, y, x_validation, y_validation):
    model.compile(
        loss='mae',
        optimizer=optimizers.Adam()
        )

    model.fit(
        x, 
        y,
        validation_data=(x_validation, y_validation),
        batch_size=3,
        epochs=1000,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.0000001, verbose=1),
            callbacks.TensorBoard(log_dir=os.path.join('logs', 'fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
            ])
            
    return model

def gradient_boost():
    train_pool = Pool(data=x_train, label=y_train)
    validation_pool = Pool(data=x_validation, label=y_validation)
    
    model = CatBoostRegressor(
        random_seed=SEED,
        iterations=1000,
        depth=6,
        learning_rate=0.8,
        loss_function='MAE',
        verbose=True,
        task_type='CPU'
        )
    
    model.fit(
        X=train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=20
        )
        
    feature_importance = model.get_feature_importance(
        data=train_pool,
        type='FeatureImportance',
        prettified=False,
        thread_count=-1,
        verbose=False
        )

    feature_importance_df = pd.DataFrame({
        'importance': feature_importance,
        'label': LABELS
        })
    feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

    p = feature_importance_df.plot(x='label', y='importance', kind='bar')
    p.set_title('Feature Importance')
    p.set_ylabel('')
    p.set_xlabel('')
    plt.savefig('feature_importance.png')

    return model

dnn = deep_neural_network()
gb = gradient_boost()
for i, x in enumerate(x_test):
    print(
        'Real prise {real}$, predicted_dnn {predicted_dnn}$, predicted_gb {predicted_gb}$ Data {data}'.format(
            real=int(y_test[i]*1000),
            predicted_dnn=int(dnn.predict(np.array([x]))[0][0]*1000),
            predicted_gb=int(gb.predict(np.array([x]))[0]*1000),
            data=pd.DataFrame(data=[x], columns=LABELS).to_dict('index')
        )
    )

evaluation_dnn = dnn.evaluate(x_test, y_test, batch_size=1)
evaluation_gb = gb.get_best_score()['validation']['MAE']

print('Evaluation DNN mae:', evaluation_dnn)
print('Evaluation GB mae:', evaluation_gb)