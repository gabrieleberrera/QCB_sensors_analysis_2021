import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = 'dataset/data/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

y_train_onehot = np.zeros((y_train.shape[0],5))


classes = np.unique(np.concatenate(Y.diagnostic_superclass.to_numpy()))

y_train_onehot = np.zeros((y_train.shape[0], len(classes)))
y_train_onehot = pd.DataFrame(y_train_onehot, columns=classes)

for i, true_list in enumerate(y_train.to_numpy()):
    print(i)
    if true_list:
        for true_label in true_list:
            y_train_onehot[true_label].iloc[i] = 1.0

y_test_onehot = np.zeros((y_test.shape[0], len(classes)))
y_test_onehot = pd.DataFrame(y_test_onehot, columns=classes)

for i, true_list in enumerate(y_test.to_numpy()):
    print(i)
    if true_list:
        for true_label in true_list:
            y_test_onehot[true_label].iloc[i] = 1.0

np.save("dataset/X_train", X_train)
y_train_onehot.to_csv("dataset/y_train.csv", index=False)

np.save("dataset/X_test", X_test)
y_test_onehot.to_csv("dataset/y_test.csv", index=False)
