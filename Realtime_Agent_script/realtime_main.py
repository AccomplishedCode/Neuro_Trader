
import argparse
import pandas as pd

import bitmex_back
from trader import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import pickle
from bitmex_back import *
import asyncio
import sys


#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--data", required=True, help="path to csv file")
#ap.add_argument("-t", "--train", required=True,  help="train the model")
#ap.add_argument("l","--load", help="load pretrained model")
#args = vars(ap.parse_args())

#df = pd.read_csv(args["data"])

def get_state(parameters, t, window_size=40):
    outside = []
    d = t - window_size + 1
    for parameter in parameters:
        block = (
            parameter[d: t + 1]
            if d >= 0
            else -d * [parameter[0]] + parameter[0: t + 1]
        )
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        for i in range(1, window_size, 1):
            res.append(block[i] - block[0])
        outside.append(res)
    return np.array(outside).reshape((1, -1))

#TODO: Confirm asyncio loop at the end from Alex

table= "tradeBin5m"
df = bitmex_back.get_raw_data(table)

#Neural Network Hyperparameters
skip = 1
layer_size = 600
output_size = 3
window_size = 40

#Account info fields
balances = client.User.User_getWalletSummary().result()
account_bal=[i['walletBalance'] for i in balances[0]]
capital=account_bal[0]

inventory_size = get_position()
mean_inventory = np.mean(inventory_size)

real_trend = df['Close'].tolist()
parameters = [df['Close'].tolist(), df['Volume'].tolist()]
concat_parameters = np.concatenate([get_state(parameters, 40), [[inventory_size, mean_inventory,capital ]]], axis=1)
input_size = concat_parameters.shape[1]
minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
initial_money = np.max(parameters[0]) * 5

#Uncomment the below code if you want to train
"""
class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.rand(input_size, layer_size)
            * np.sqrt(1 / (input_size + layer_size)),
            np.random.rand(layer_size, output_size)
            * np.sqrt(1 / (layer_size + output_size)),
            np.zeros((1, layer_size)),
            np.zeros((1, output_size)),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-2]
        decision = np.dot(feed, self.weights[1]) + self.weights[-1]
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
"""

# Load model from disk

with open('model2.pkl', 'rb') as fopen:
    model = pickle.load(fopen)

#model = Model(input_size=input_size, layer_size=layer_size, output_size=output_size)

# Redefine update function
"""
def update(self, table="tradeBin5m", force_update=True):
    self.reupdate= False
    data = get_raw_data(table)
    work_state = self.params['work_state']
    try:
        row= data.tail(1 +self.params['period'])
        work_state_current= row[work_state].values[0]
        if self.state_open:
            self.update_wait2(work_state_current, force_update= force_update)
        else:
            self.update_wait(work_state_current, force_update= force_update)
    except Exception as e:
        self.reupdate=True
        print(e)
        print(sys.exc_info()[2])

"""
async def start():
    agent = Agent(model=model, timeseries=scaled_parameters, skip=skip, initial_money=capital, real_trend=real_trend,
                  minmax=minmax)
    bitmex_back.start(True)

    while True:
        await asyncio.sleep(5)
        agent.buy()
        #trader.update(force_update= trader.reupdate)

if __name__== "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(start())




"""
def fit_func(agent):
    print("starting training....")
    agent.fit(iterations=200, checkpoint=1)
    print()

fit_func(agent)

"""