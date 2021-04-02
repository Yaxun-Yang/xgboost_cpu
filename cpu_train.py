import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost import  XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import os


def data_process():
    root_dir_str = './data/input/'
    root_dir = os.path.join(root_dir_str)
    out_data = pd.read_csv("./data/output.csv", usecols=[1, 2]).to_numpy()
    # print(out_data)
    data = []
    feature_name = []

    for (dir_paths, dir_names, file_names) in os.walk(root_dir):
        for file_name in file_names:
            in_data = pd.read_csv(root_dir_str + file_name, usecols=[1, 2], nrows=1).to_numpy()
            while out_data[0][0] < in_data[0][0]:
                out_data = np.delete(out_data, 0, axis=0)

    for (dir_paths, dir_names, file_names) in os.walk(root_dir):
        for file_name in file_names:
            feature_name.append(file_name.split(".")[0])
            in_data = pd.read_csv(root_dir_str + file_name, usecols=[1, 2]).to_numpy()
            temp_data = []
            i = 0
            for a_data in out_data:
                while True:
                    if i == len(in_data) - 1:
                        temp_data.append(in_data[i][1])
                        break
                    if in_data[i+1][0] > a_data[0] and in_data[i][0] < a_data[0]:
                        temp_data.append((in_data[i+1][1]+in_data[i][1])/2)
                        break
                    else:
                        i += 1

            data.append(temp_data)

    data = np.array(data)
    print(data.shape)
    data = np.transpose(data)
    df = pd.DataFrame(data)
    df.to_csv("data/train.csv", mode='w', sep=",", index=False, header=feature_name)
    df = pd.DataFrame(np.transpose(out_data)[1])
    df.to_csv("data/label.csv", mode='w', sep=",", index=False)


def read_data():
    x_train_data = pd.read_csv("data/train.csv",).to_numpy()

    feature_name = pd.read_csv("data/train.csv", nrows=1, header=None).to_numpy()[0].tolist()
    print(feature_name)

    y_train_data = pd.read_csv("data/label.csv").to_numpy()

    x_train_data, x_test_data, y_train_data, y_test_data = \
        train_test_split(x_train_data, y_train_data, test_size=0.2, random_state=0)

    dtrain = xgb.DMatrix(x_train_data, label=y_train_data, feature_names=feature_name)
    dtest = xgb.DMatrix(x_test_data, label=y_test_data, feature_names = feature_name)
    return dtrain, dtest


def train():
    dtrain, dtest= read_data()
    # # learning_rate, min_split_split_loss, , reg_lambda, reg_alpha
    # param = {"eta": 0.1, "gamma": 0, "max_depth": 6, "lambda": 1, "alpha": 0, "objective": "reg:gamma"}
    # num_round = 200
    # model = xgb.train(param, dtrain, num_round)
    # model.save_model('data/xgb.model')
    model = xgb.Booster(model_file='data/xgb.model')
    pred = model.predict(dtest)
    print(pred)
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] =True
    xgb.plot_importance(model, title="重要程度前十的特征", xlabel="得分", ylabel="特征", grid=False, max_num_features=10)
    # xgb.plot_tree(model, fmap='', num_trees=6, rankdir=' UT', ax=None)
    plt.show()


def main():

    # data_process()
    train()


if __name__ == '__main__':
    main()
