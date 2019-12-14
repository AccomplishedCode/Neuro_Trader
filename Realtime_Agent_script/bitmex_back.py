import datetime
import threading
from enum import auto
from time import sleep
import os
import bitmex
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import json
import asyncio


from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels, BaseChannels

BITMEX_API_KEY =''
BITMEX_API_SECRET = ''

client = bitmex.bitmex(test=True, api_key=BITMEX_API_KEY, api_secret=BITMEX_API_SECRET)

bins = {}
max_len = 40

OnUpdate = None



class InstrumentChannels2(BaseChannels):
    tradeBin1m = auto()
    tradeBin5m = auto()
    tradeBin1h = auto()
    tradeBin1d = auto()


def get_position(symbol='XBTUSD'):
    return client.Position.Position_get(filter=json.dumps({'symbol': symbol})).result()

def place_order(price=None, type="Market", qty=1, stopPx=None, execInst = None):
    return client.Order.Order_new(symbol='XBTUSD', orderQty=qty, price=price, ordType=type, stopPx=stopPx, execInst=execInst).result()
    # client.Order.Order_new(symbol='XBTUSD', orderQty=-10, price=12345.0).result()

def edit_order(price, orderID):
    return client.Order.Order_amend(orderID=orderID, price=price).result()

def cancel_all():
    client.Order.Order_cancelAll().result()

def get_orders(count = 10):
    return client.Order.Order_getOrders(reverse=True, count=count).result()


def add_bins(table, data, update = False):

    df = pd.DataFrame(data=data, columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'trades', 'volume'])
    df = df[['timestamp', 'open', 'close', 'high', 'low', 'trades', 'volume']]
    df.columns = ["Date", "Open", "Close", "High", "Low", "Trades", "Volume"]

    df['Datetime'] = df['Date'].apply(lambda x: pd.to_datetime(x).tz_localize(None) if isinstance(x,
                                                                                                  datetime.datetime) else datetime.datetime.strptime(
        x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    df['Date'] = df['Datetime'].apply(lambda x: x.date())
    update = True
    if table in bins:
        #
        newData = bins[table].shape[0] == 0 or \
                  df.shape[0] == 0 or \
                  bins[table]['Datetime'].values[-1] != df['Datetime'].values[-1]
        if newData:
            #if not table == 'tradeBin1m':
            #    print("Append table to bin", table)

            df = bins[table].append(df, ignore_index=True, sort=False)
            df.sort_values(by='Datetime', axis=0, ascending=True, inplace=True)
            df = df.drop_duplicates(subset='Datetime')
            bins[table] = df
        else: # nothing new in the data
            print("No new data at", table)
            return
    else:
        df.sort_values(by='Datetime', axis=0, ascending=True, inplace=True)
        bins[table] = df

    if update:
        if not OnUpdate is None:
            OnUpdate(table)
    else:
        print("Init bin", table)


def get_raw_data(table):
    data = bins.get(table, pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Trades", "Volume"]))
    return data


def on_message(message):
    table = message['table']

    data = message['data']
    if table == 'trade':
        return
    elif table == 'orderBook10':
        return
    elif table == 'instrument':
        return
    elif table == 'order':
        return
    else:

        add_bins(table, data, True)


def start(thread = False):
    channels = [
        InstrumentChannels2.tradeBin5m,
    ]

    print("load trade history")
    res = client.Trade.Trade_getBucketed(symbol='XBTUSD', binSize="5m", reverse=True, count=750).result()
    add_bins("tradeBin{}".format("5m"), res[0])

    XBTUSD = Instrument(symbol='XBTUSD', channels=channels, should_auth=False)
    XBTUSD.on('action', on_message)
    #wst = None

    def run_loop(c):
        print("Run loop")
        while True:
            try:
                XBTUSD.run_forever()
            except Exception as e:
                print(e)
                sleep(1)
                pass

    if thread:
        print("Loop with thread")
        wst = threading.Thread(target=lambda: run_loop(XBTUSD.run_forever))
        wst.daemon = True
        wst.start()
    else:
        run_loop(None)

if __name__=="__main__":
    start()

