# Pycharm
# What's cool: right click on a variable then "refactor" to rename everywhere
# Ctrl + Shift + Up/Down to move statement
# Ctrl + F for find
# Split window in 2 to have view while modifying the file
# Ctrl + D to duplicate rows
# # todo write_smth_todo - then display the todos in down palette
# Alt + Scroll = select only the column
# Surround with: click on variable > Ctrl + Alt + T

# Debugging
# Assess variable states
# Assert new expressions: + new watch | works well with stop
# Exec pas à pas: step into = opens new tab with ref method
# Step over / other options: play with them.
# Evaluate expressions: you're able to evaluate any expression with the variables in scope
# Returns exact value + type
# bin () doesn't go backward.

import requests
import json
#import pandas as pd
import glob
import datetime
import time

# Returns actual volume weighted price
# Exchange rate
# Prices and volumes across well connected exchanges
'''
url = 'https://min-api.cryptocompare.com/data/dayAvg?fsym=BTC&tsym=USD&avgType=VolFVolT&toTs=1550668169&extraParams=research&api_key={afb994efa0ca3fc08f6b4b4b1ce74d3cfdfbf962b4c2e6fe4a5694dd37cd68aa}'
response = requests.get(url, headers = {"User-Agent":"Mozilla"}).json()
start = datetime.datetime.strptime("02-05-2019", "%d-%m-%Y")
end = datetime.datetime.strptime("01-10-2019", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_generated_unix = [int(x.timestamp()) for x in date_generated]

ccs = ['BTC', 'EOS', 'XRP', 'LTC', 'ETH']

def gen_json(curr:str, date_list:list):
    repo = []
    for epoch in date_list:
        url = 'https://min-api.cryptocompare.com/data/dayAvg?fsym='+curr+'&tsym=USD&avgType=VolFVolT&toTs='+str(epoch)+'&extraParams=research&api_key={afb994efa0ca3fc08f6b4b4b1ce74d3cfdfbf962b4c2e6fe4a5694dd37cd68aa}'
        repo.append(requests.get(url, headers={"User-Agent": "Mozilla"}).json())
        time.sleep(3)
    return repo

twaps = {}
for cc in ccs:
    twaps[cc] = pd.DataFrame(gen_json(cc, date_generated_unix))
'''