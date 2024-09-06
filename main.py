import yfinance as yf
from datetime import datetime, timedelta
from pprint import pformat
import json
import logging
from mknapsack import solve_unbounded_knapsack
import mknapsack._exceptions as mknapsack_exceptions
from argparse import ArgumentParser
import traceback
import pandas
import multiprocessing
import yahooquery
from numpy import nan


parser = ArgumentParser()
parser.add_argument("-a", "--anchor", help="Anchor ticker")
parser.add_argument(
    "-C", "--refresh-cache", help="Force refresh cache", action="store_true"
)
parser.add_argument(
    "-M", "--skip-cache-write", help="Skip cache write", action="store_true"
)
parser.add_argument("-i", "--total-invest-available", help="Total invest available")
parser.add_argument("--verbose", "-v", action="count", default=0)
args = parser.parse_args()

if args.verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif args.verbose > 1:
    logging.basicConfig(level=logging.DEBUG)

anchor_ticker = args.anchor
total_invest_available = float(args.total_invest_available)

stdev_max = 99
return_field = "fiveYearAverageReturn"
correlation_field = "Close"

ticker_cache = None

try:
    if args.refresh_cache:
        ticker_cache = multiprocessing.Manager().dict()
    else:
        with open("ticker_cache.json") as f:
            ticker_cache = json.loads(f.read())

except:  # noqa: E722
    logging.error(traceback.format_exc())
    ticker_cache = multiprocessing.Manager().dict()


def download_stock_data(ticker_name):

    logging.info("Download " + ticker_name)
    ticker = yf.Ticker(ticker_name)
    yq_ticker = yahooquery.Ticker(ticker_name)
    result = ticker.info | {
        "history": json.loads(ticker.history(period="5y", interval="1d").to_json())
    }
    logging.debug(ticker_name)
    logging.debug(yq_ticker.fund_profile[ticker_name])
    try:
        result.update(
            {
                "netExpenseRatio": yq_ticker.fund_profile[ticker_name]
                .get("feesExpensesInvestment")
                .get("annualReportExpenseRatio")
            }
        )
        # result.update({"netExpenseRatio": 1})
    except AttributeError:
        result.update({"netExpenseRatio": None})

    import copy

    ticker_cache[ticker_name] = copy.deepcopy(result)


with open("tickers.txt") as f:
    ticker_names = []
    for ticker_name in f:
        ticker_name = ticker_name.strip()
        if not ticker_name.startswith("#") and ticker_cache.get(ticker_name) is None:
            ticker_names += [ticker_name]
    with multiprocessing.Pool(processes=48) as p:
        p.map(download_stock_data, ticker_names)
    ticker_cache = dict(ticker_cache)


def set_security_price_field(fields_to_attempt: list):
    while len(fields_to_attempt) > 0:
        field = fields_to_attempt.pop(0)
        # if None in [x.get(field) for _, x in ticker_cache.items()]:
        if ticker_cache[anchor_ticker].get(field) is None:
            logging.warning(
                "some tickers missing '%s' field, trying different field", field
            )
            continue

        else:
            logging.info("using '%s' field for security price", field)
            return field


security_price_field = set_security_price_field(["ask", "close", "open", "navPrice"])
# security_price_field = set_security_price_field(["ask"])


# calculate stdev between two numbers
def stdev(a, b):
    return round((b - a) ** 2, 8)


logging.debug(pformat(ticker_cache[anchor_ticker]))

corr_query = [
    (
        ticker_name,
        pandas.DataFrame.from_dict(info.get("history")).corrwith(
            pandas.DataFrame.from_dict(ticker_cache[anchor_ticker]["history"])
        ),
    )
    for ticker_name, info in ticker_cache.items()
]
corr_query = sorted(corr_query, key=lambda x: x[1].to_dict()[correlation_field])
for x in corr_query:
    logging.debug("%s %f", x[0], x[1].to_dict()[correlation_field])
    # ticker_cache[x[0]]["correlation"] = x[1].to_dict()
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
    # if info.get("correlationFactor") is not None
]
stdev_query = sorted(stdev_query, key=lambda x: x[1], reverse=True)
for x in stdev_query:
    ticker_cache[x[0]]["stdev"] = x[1]
logging.debug(pformat(stdev_query))

SHIFT = 10**4
variance_max = stdev_max**2
logging.debug(pformat(ticker_cache[anchor_ticker]))

total_invest_minus_target = (
    total_invest_available
    - int(total_invest_available / ticker_cache[anchor_ticker][security_price_field])
    * ticker_cache[anchor_ticker][security_price_field]
)
logging.info(total_invest_minus_target)

# select tickers to filter here
valid_tickers = {}
for ticker_name, x in ticker_cache.items():
    if (
        x.get(security_price_field)
        and x[security_price_field] < total_invest_minus_target
        and x.get("stdev")
        and (
            x["stdev"]
            < variance_max
            #            or x[return_field]
            #            > ticker_cache[anchor_ticker][return_field]
        )
        and datetime.now()
        - datetime.fromtimestamp(x.get("fundInceptionDate", datetime.now().timestamp()))
        > timedelta(weeks=52 * 5)
        and x.get("correlationFactor") != nan
        # and x.get("netExpenseRatio") < ticker_cache[anchor_ticker].get("netExpenseRatio")
    ):
        logging.debug(x["stdev"])
        valid_tickers[ticker_name] = x

result = None
try:
    profits = [
        int((x["stdev"] / 10) ** -1)
        # int((x["stdev"] / x["netExpenseRatio"]) ** -1 * SHIFT)
        for ticker_name, x in valid_tickers.items()
    ]
    weights = [
        int(x[security_price_field] * SHIFT) for ticker_name, x in valid_tickers.items()
    ]
    n_items = [
        int(total_invest_minus_target / x[security_price_field])
        for ticker_name, x in valid_tickers.items()
    ]
    capacity = int(total_invest_minus_target * SHIFT)
    logging.debug(pformat([profits, weights, n_items, capacity]))
    ks_verbose = True if logging.getLogger().level == logging.DEBUG else False
    # result = solve_bounded_knapsack( profits, weights, n_items, capacity, verbose=ks_verbose)
    result = solve_unbounded_knapsack(profits, weights, capacity, verbose=ks_verbose)
    # result = knapsack(profits, weights).solve(capacity)
except mknapsack_exceptions.FortranInputCheckError as e:
    if e.z == -1:
        result = [0] * len(valid_tickers)
    else:
        raise e


logging.debug(list(result))

total_invest_calculated = 0
i = 0
print(
    "tickername",
    "count",
    security_price_field,
    return_field,
    "netExpenseRatio",
    "correlationFactor",
    "stdev",
)
print(
    anchor_ticker,
    int(total_invest_available / ticker_cache[anchor_ticker][security_price_field]),
    ticker_cache[anchor_ticker][security_price_field],
    ticker_cache[anchor_ticker].get(return_field),
    ticker_cache[anchor_ticker].get("netExpenseRatio"),
    ticker_cache[anchor_ticker]["correlationFactor"],
    "{:.8f}".format(ticker_cache[anchor_ticker]["stdev"]),
)
total_invest_calculated += (
    int(total_invest_available / ticker_cache[anchor_ticker][security_price_field])
    * ticker_cache[anchor_ticker][security_price_field]
)
for ticker_name in [ticker_name for ticker_name, x in valid_tickers.items()]:
    if result[i] > 0:
        logging.debug("total_invest_calculated %f", total_invest_calculated)
        print(
            ticker_name,
            result[i],
            ticker_cache[ticker_name][security_price_field],
            ticker_cache[ticker_name].get(return_field),
            ticker_cache[ticker_name].get("netExpenseRatio"),
            ticker_cache[ticker_name]["correlationFactor"],
            "{:.8f}".format(ticker_cache[ticker_name]["stdev"]),
        )
        total_invest_calculated += (
            result[i] * ticker_cache[ticker_name][security_price_field]
        )
    i += 1
print(total_invest_calculated)

if not args.skip_cache_write:
    try:
        with open("ticker_cache.json", "w") as f:
            payload = json.dumps(ticker_cache, indent=4)
            f.write(payload)
    except:  # noqa: E722
        logging.error(traceback.format_exc())
        logging.error("couldn't write cache")
