# poorboy

Given a specific security, calculate other similar security buys constrained by a budget.

```
$ python3 main.py VOO 200 
tickername,count,ask,fiveYearAverageReturn,netExpenseRatio,correlationFactor,stdev
VOO,0,495.47,0.1447936,0.00029999999,1.0,0.00000000
SPTM,2,67.19,0.14204541,0.00029999999,0.9996371336547825,0.00000013
SPXV,1,57.36,0.1452651,0.00090000004,0.9995288466940137,0.00000022

```

## `tickers.txt`

`tickers.txt` is a user-provided file must be filled with a newline separated list of tickers to match against. It is the user's responsibility to source a list of securities. Generally, the larger the list, the higher quality the results from `poorboy` will be.

## `ticker_cache.json`

The `ticker_cache.json` file is generated by the program and contains information about each ticker in the `tickers.txt` file. It is used to speed up subsequent runs of `poorboy`. The file can get quite large, with about 300kB per ticker.