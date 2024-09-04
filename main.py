import yfinance as yf
from datetime import datetime, timedelta
from pprint import pprint
import json
import logging
from mknapsack import solve_bounded_knapsack
import mknapsack._exceptions as mknapsack_exceptions
from argparse import ArgumentParser
import traceback
import pandas

# logging.basicConfig(level=logging.INFO)

p = ArgumentParser()
p.add_argument("-a", "--anchor", help="Anchor ticker")
p.add_argument("-C", "--refresh-cache", help="Force refresh cache", action="store_true")
p.add_argument("-i", "--total-invest-available", help="Total invest available")
args = p.parse_args()

anchor_ticker = args.anchor
total_invest_available = float(args.total_invest_available)

stdev_max = 0.02
return_field = "fiveYearAverageReturn"
share_price_field = "open"
correlation_field = "Close"

ticker_cache = None

try:
    if args.refresh_cache:
        ticker_cache = {}
    else:
        with open("ticker_cache.json") as f:
            ticker_cache = json.loads(f.read())

except:
    logging.error(traceback.format_exc())
    ticker_cache = {}


def download_stock_data(ticker_name):

    logging.info("Download " + ticker_name)
    ticker = yf.Ticker(ticker_name)
    result = ticker.info | {
        "history": json.loads(ticker.history(period="5y", interval="1d").to_json())
    }

    return result


with open("tickers.txt") as f:
    for ticker_name in f:
        ticker_name = ticker_name.strip()
        if not ticker_name.startswith("#") and ticker_cache.get(ticker_name) is None:
            ticker_cache[ticker_name] = download_stock_data(ticker_name)


# calculate stdev between two numbers
def stdev(a, b):
    return (b - a) ** 2


# pprint(ticker_cache[anchor_ticker])

corr_query = [
    (
        ticker_name,
        pandas.DataFrame.from_dict(info.get("history")).corrwith(
            pandas.DataFrame.from_dict(ticker_cache[anchor_ticker]["history"])
        ),
    )
    for ticker_name, info in ticker_cache.items()
]
# corr_query = sorted(corr_query, key=lambda x: x[1], reverse=True)
for x in corr_query:
    # print(x[0], x[1])
    ticker_cache[x[0]]["correlation"] = x[1].to_dict()
    ticker_cache[x[0]]["correlationFactor"] = x[1].to_dict()[correlation_field]

stdev_query = [
    (
        ticker_name,
        stdev(
            ticker_cache[anchor_ticker]["correlationFactor"],
            info.get("correlationFactor", 0),
        ),
    )
    for ticker_name, info in ticker_cache.items()
    if info.get("correlationFactor") is not None
]
stdev_query = sorted(stdev_query, key=lambda x: x[1], reverse=True)
for x in stdev_query:
    ticker_cache[x[0]]["stdev"] = x[1]
# pprint(stdev_query)

SHIFT = 10**4
variance_max = stdev_max**2
# print(ticker_cache[anchor_ticker])

total_invest_minus_target = (
    total_invest_available
    - int(total_invest_available / ticker_cache[anchor_ticker][share_price_field])
    * ticker_cache[anchor_ticker][share_price_field]
)

# select tickers to filter here
valid_tickers = {}
for ticker_name, x in ticker_cache.items():
    if (
        x.get(share_price_field)
        and x[share_price_field] < total_invest_minus_target
        and x.get("stdev")
        and (
            x["stdev"]
            < variance_max
            #            or x[return_field]
            #            > ticker_cache[anchor_ticker][return_field]
        )
        and datetime.now() - datetime.fromtimestamp(x["fundInceptionDate"])
        > timedelta(weeks=52 * 5)
    ):
        # print(x["stdev"])
        valid_tickers[ticker_name] = x

result = None
try:
    profits = [int(x["stdev"] ** -1) for ticker_name, x in valid_tickers.items()]
    weights = [
        int(x[share_price_field]) * SHIFT for ticker_name, x in valid_tickers.items()
    ]
    n_items = [
        int(total_invest_minus_target / x[share_price_field])
        for ticker_name, x in valid_tickers.items()
    ]
    capacity = int(total_invest_minus_target) * SHIFT
    # print(profits, weights, n_items, capacity)
    result = solve_bounded_knapsack(profits, weights, n_items, capacity)
except mknapsack_exceptions.FortranInputCheckError as e:
    if e.z == -1:
        result = [0] * len(valid_tickers)
    else:
        raise e


print(list(result))

i = 0
print("tickername", "count", share_price_field, return_field, "correlationFactor")
print(
    anchor_ticker,
    int(total_invest_available / ticker_cache[anchor_ticker][share_price_field]),
    ticker_cache[anchor_ticker][share_price_field],
    ticker_cache[anchor_ticker].get(return_field),
    ticker_cache[anchor_ticker]["correlationFactor"],
)
for ticker_name in [ticker_name for ticker_name, x in valid_tickers.items()]:
    if result[i] > 0:
        print(
            ticker_name,
            result[i],
            ticker_cache[ticker_name][share_price_field],
            ticker_cache[ticker_name].get(return_field),
            ticker_cache[ticker_name]["correlationFactor"],
        )
        total_invest_minus_target -= (
            result[i] * ticker_cache[ticker_name][share_price_field]
        )
    i += 1
print(total_invest_minus_target)

try:
    with open("ticker_cache.json", "w") as f:
        payload = json.dumps(ticker_cache, indent=4)
        f.write(payload)
except:
    logging.error("couldn't write cache")
