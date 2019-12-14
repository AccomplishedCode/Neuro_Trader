import numpy as np

import bitmex_back
from strategy import Strategy
from bitmex_back import *


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

balances = client.User.User_getWalletSummary().result()
account_bal=[i['walletBalance'] for i in balances[0]]
capital=account_bal[0]
window_size=40

class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.5
    LEARNING_RATE = 0.3

    def __init__(self, model, timeseries, skip, initial_money, real_trend, minmax):
        self.model = model
        self.timeseries = timeseries
        self.skip = skip
        self.real_trend = real_trend
        self.initial_money = initial_money
        self.es = Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )
        self.minmax = minmax
        self._initiate()

    def _initiate(self):
        # i assume first index is the close value
        self.trend = self.timeseries[0]
        self._mean = np.mean(self.trend)
        self._std = np.std(self.trend)
        self._inventory = []
        self._capital = self.initial_money
        self._queue = []
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        self._queue = []
        self._inventory = []

    def change_data(self, timeseries, skip, initial_money, real_trend, minmax):
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self.minmax = minmax
        self._initiate()

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0])

    def act_softmax(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0]), softmax(decision)[0]

    def get_state(self, t, inventory, capital, timeseries):
        state = get_state(timeseries, t)
        len_inventory = len(inventory)
        if len_inventory:
            mean_inventory = np.mean(inventory)
        else:
            mean_inventory = 0
        z_inventory = (mean_inventory - self._mean) / self._std
        z_capital = (capital - self._mean) / self._std
        concat_parameters = np.concatenate(
            [state, [[len_inventory, z_inventory, z_capital]]], axis=1
        )
        return concat_parameters

    def get_reward(self, weights):
        initial_money = self._scaled_capital
        starting_money = initial_money
        invests = []
        self.model.weights = weights
        inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t]

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += self.trend[t]
                invest = ((self.trend[t] - bought_price) / bought_price) * 100
                invests.append(invest)

            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )
        invests = np.mean(invests)
        if np.isnan(invests):
            invests = 0
        score = (starting_money - initial_money) / initial_money * 100
        return invests * 0.7 + score * 0.3

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):

        initial_money = self._scaled_capital
        starting_money = initial_money
        inventory = bitmex_back.get_position('XBTUSD')
        state = self.get_state(0, inventory, starting_money, self.timeseries)
        fees=0
        states_sell = 0
        states_buy = 0

        for t in range(0, len(self.trend) - 1, self.skip):
            action, prob = self.act_softmax(state)
            print(t, prob)

            if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - 1 - window_size):
                bitmex_back.place_order(price=self.real_trend[t], type="Market", qty=1, stopPx=self.real_trend[t] * 1.03)
                updated_balance = [i['walletBalance'] for i in balances[-1]][-1]
                timestamp= [i['timestamp'] for i in balances[-1]][-1]
                fee_for_trade=[i['fee'] for i in balances[-1]][-1]
                fees += fee_for_trade
                states_buy +=1
                print(
                    'day %d: buy 1 unit at price %f, total balance %f'
                    % (timestamp, self.real_trend[t], updated_balance)
                )

            elif action == 2 and len(inventory):
                bitmex_back.place_order(price=self.real_trend[t], type="Market", qty=-1, stopPx=self.real_trend[t] * 0.97)
                updated_balance = [i['walletBalance'] for i in balances[-1]][-1]
                investment_updated = [i['unrealizedPnL'] for i in balances[-1]]
                investment = investment_updated[0]
                timestamp = [i['timestamp'] for i in balances[-1]][-1]
                fee_for_trade = [i['fee'] for i in balances[-1]][-1]
                fees += fee_for_trade
                states_sell +=1

                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (timestamp, self.real_trend[t], investment, updated_balance)
                )
            state = self.get_state(t + 1, inventory, starting_money, self.timeseries)
        unrealized_gains= [i['unrealizedPnL'] for i in balances[-1]][-1]
        realized_gains= [i['realizedPnL'] for i in balances[-1]][-1]

        total_gains=unrealized_gains +realized_gains - fees
        return states_buy, states_sell, total_gains, fees
"""
    def update(self, table="tradeBin5m", force_update=True):
        self.reupdate= False
        data= driver.get_raw_data(table)
        work_state= self.params["work_state"]

        try:
            row=data.tail(1+self.params['period'])
            work_state_current= row[work_state].values[0]

            if self.state_open:
                self.update_wait
"""

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
