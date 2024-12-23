from pprint import pformat
import yfinance as yf
from datetime import datetime, timedelta
import json
import traceback
import multiprocessing
import yahooquery
from numpy import nan
from pandas import DataFrame
import logging

RETURN_FIELD = "fiveYearAverageReturn"
CORRELATION_FIELD = "Close"

logger = logging.getLogger("poorboy")


def solve_unbounded_knapsack(
    profits: list[int], weights: list[int], capacity: int, verbose=False
) -> list[int]:
    # Initialize a table to store the maximum value for each subproblem
    dp = [[0] * (capacity + 1) for _ in range(len(profits) + 1)]

    # Fill the table using dynamic programming
    for i in range(1, len(profits) + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j:
                dp[i][j] = max(
                    dp[i - 1][j],
                    profits[i - 1] * (j // weights[i - 1])
                    + dp[i - 1][j % weights[i - 1]],
                )
            else:
                dp[i][j] = dp[i - 1][j]

    # Backtrack to find the quantities of each object
    quantities = [0] * len(profits)
    j = capacity
    for i in range(len(profits), 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            quantity = (dp[i][j] - dp[i - 1][j]) // profits[i - 1]
            quantities[i - 1] += quantity
            j -= weights[i - 1] * quantity

    return quantities


class Poorboy:

    ticker_cache = None

    def download_stock_data(self, ticker_name):

        logger.info("Download " + ticker_name)
        ticker = yf.Ticker(ticker_name)
        yq_ticker = yahooquery.Ticker(ticker_name)
        result = ticker.info | {
            "history": json.loads(ticker.history(period="5y", interval="1d").to_json())
        }
        logger.debug(ticker_name)
        logger.debug(yq_ticker.fund_profile[ticker_name])
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

        self.ticker_cache[ticker_name] = copy.deepcopy(result)

    def calculate_security_buys(
        self,
        anchor_ticker: str,
        total_invest_available: float,
        force_refresh_cache=False,
        skip_cache_write=False,
        stdev_max=99,
    ) -> DataFrame:

        try:
            if force_refresh_cache:
                self.ticker_cache = multiprocessing.Manager().dict()
            else:
                with open("ticker_cache.json") as f:
                    self.ticker_cache = multiprocessing.Manager().dict(
                        json.loads(f.read())
                    )

        except:  # noqa: E722
            logger.error(traceback.format_exc())
            self.ticker_cache = multiprocessing.Manager().dict()

        with open("tickers.txt") as f:
            ticker_names = []
            for ticker_name in f:
                ticker_name = ticker_name.strip()
                if (
                    not ticker_name.startswith("#")
                    and self.ticker_cache.get(ticker_name) is None
                ):
                    ticker_names += [ticker_name]
            with multiprocessing.Pool(processes=12) as p:
                p.map(self.download_stock_data, ticker_names)
            self.ticker_cache = dict(self.ticker_cache)

        logger.debug(sorted(self.ticker_cache.keys()))

        def set_security_price_field(fields_to_attempt: list):
            while len(fields_to_attempt) > 0:
                field = fields_to_attempt.pop(0)
                # if None in [x.get(field) for _, x in self.ticker_cache.items()]:
                if self.ticker_cache[anchor_ticker].get(field) is None:
                    logger.warning(
                        "some tickers missing '%s' field, trying different field", field
                    )
                    continue

                else:
                    logger.info("using '%s' field for security price", field)
                    return field

        security_price_field = set_security_price_field(
            ["ask", "close", "open", "navPrice"]
        )
        # security_price_field = set_security_price_field(["ask"])

        # calculate stdev between two numbers
        def stdev(a, b):
            return round((b - a) ** 2, 8)

        logger.debug(pformat(self.ticker_cache[anchor_ticker]))

        corr_query = [
            (
                ticker_name,
                DataFrame.from_dict(info.get("history")).corrwith(
                    DataFrame.from_dict(self.ticker_cache[anchor_ticker]["history"])
                ),
            )
            for ticker_name, info in self.ticker_cache.items()
        ]
        corr_query = sorted(corr_query, key=lambda x: x[1].to_dict()[CORRELATION_FIELD])
        for x in corr_query:
            logger.debug("%s %f", x[0], x[1].to_dict()[CORRELATION_FIELD])
            # self.ticker_cache[x[0]]["correlation"] = x[1].to_dict()
            self.ticker_cache[x[0]]["correlationFactor"] = x[1].to_dict()[
                CORRELATION_FIELD
            ]

        stdev_query = [
            (
                ticker_name,
                stdev(
                    self.ticker_cache[anchor_ticker]["correlationFactor"],
                    info.get("correlationFactor", 0),
                ),
            )
            for ticker_name, info in self.ticker_cache.items()
            # if info.get("correlationFactor") is not None
        ]
        stdev_query = sorted(stdev_query, key=lambda x: x[1], reverse=True)
        for x in stdev_query:
            self.ticker_cache[x[0]]["stdev"] = x[1]
        logger.debug(pformat(stdev_query))

        if not skip_cache_write:
            try:
                with open("ticker_cache.json", "w") as f:
                    payload = json.dumps(self.ticker_cache, indent=4)
                    f.write(payload)
            except:  # noqa: E722
                logger.error(traceback.format_exc())
                logger.error("couldn't write cache")

        SHIFT = 10**2
        variance_max = stdev_max**2
        logger.debug(pformat(self.ticker_cache[anchor_ticker]))

        total_invest_minus_anchor = (
            total_invest_available
            - int(
                total_invest_available
                / self.ticker_cache[anchor_ticker][security_price_field]
            )
            * self.ticker_cache[anchor_ticker][security_price_field]
        )
        logger.info("total_invest_minus_anchor %f", total_invest_minus_anchor)

        # select tickers to filter here
        valid_tickers = {}
        for ticker_name, x in self.ticker_cache.items():
            if (
                x.get(security_price_field)
                and x[security_price_field] < total_invest_minus_anchor
                and x.get("stdev")
                and (
                    x["stdev"]
                    < variance_max
                    #            or x[return_field]
                    #            > self.ticker_cache[anchor_ticker][return_field]
                )
                and datetime.now()
                - datetime.fromtimestamp(
                    x.get("fundInceptionDate", datetime.now().timestamp())
                )
                > timedelta(weeks=52 * 5)
                and x.get("correlationFactor") != nan
                # and x.get("netExpenseRatio") < self.ticker_cache[anchor_ticker].get("netExpenseRatio")
            ):
                logger.debug(x["stdev"])
                valid_tickers[ticker_name] = x

        result = None
        try:
            profits = [
                int((x["stdev"] / SHIFT) ** -1)
                # int((x["stdev"] / x["netExpenseRatio"]) ** -1 * SHIFT)
                for ticker_name, x in valid_tickers.items()
            ]
            weights = [
                int(x[security_price_field] * SHIFT)
                for ticker_name, x in valid_tickers.items()
            ]
            n_items = [
                int(total_invest_minus_anchor / x[security_price_field])
                for ticker_name, x in valid_tickers.items()
            ]
            capacity = int(total_invest_minus_anchor * SHIFT)
            logger.debug(pformat([profits, weights, n_items, capacity]))
            ks_verbose = True if logger.level == logging.DEBUG else False
            # result = solve_bounded_knapsack( profits, weights, n_items, capacity, verbose=ks_verbose)
            result = solve_unbounded_knapsack(
                profits, weights, capacity, verbose=ks_verbose
            )
            # result = knapsack(profits, weights).solve(capacity)
        except Exception as e:
            raise e

        logger.debug(list(result))

        total_invest_calculated = 0
        i = 0
        result_df_dict = {
            "tickername": [],
            "count": [],
            security_price_field: [],
            RETURN_FIELD: [],
            "netExpenseRatio": [],
            "correlationFactor": [],
            "stdev": [],
        }

        def add_record_to_df_dict(df_dict, ticker_name, count):
            df_dict["tickername"] += [ticker_name]
            df_dict["count"] += [count]
            df_dict[security_price_field] += [
                self.ticker_cache[ticker_name][security_price_field]
            ]
            df_dict[RETURN_FIELD] += [self.ticker_cache[ticker_name].get(RETURN_FIELD)]
            df_dict["netExpenseRatio"] += [
                self.ticker_cache[ticker_name].get("netExpenseRatio")
            ]
            df_dict["correlationFactor"] += [
                self.ticker_cache[ticker_name]["correlationFactor"]
            ]
            df_dict["stdev"] += [
                "{:.8f}".format(self.ticker_cache[ticker_name]["stdev"])
            ]

        add_record_to_df_dict(
            result_df_dict,
            anchor_ticker,
            int(
                total_invest_available
                / self.ticker_cache[anchor_ticker][security_price_field]
            ),
        )
        total_invest_calculated += (
            int(
                total_invest_available
                / self.ticker_cache[anchor_ticker][security_price_field]
            )
            * self.ticker_cache[anchor_ticker][security_price_field]
        )
        for ticker_name in [ticker_name for ticker_name, x in valid_tickers.items()]:
            if result[i] > 0:
                logger.debug("total_invest_calculated %f", total_invest_calculated)
                add_record_to_df_dict(result_df_dict, ticker_name, result[i])
                total_invest_calculated += (
                    result[i] * self.ticker_cache[ticker_name][security_price_field]
                )
            i += 1
        result_df = DataFrame(result_df_dict)
        logger.info("total_invest_calculated %f", total_invest_calculated)
        logger.info("cash_remaining %f", total_invest_available - total_invest_calculated)
        return result_df
