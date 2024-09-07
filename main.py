import logging
from argparse import ArgumentParser
from poorboy import Poorboy

parser = ArgumentParser()
parser.add_argument("anchor_ticker", help="Anchor ticker")
parser.add_argument("total_invest_available", help="Total invest available")
parser.add_argument(
    "-C", "--force-refresh-cache", help="Force refresh cache", action="store_true"
)
parser.add_argument(
    "-M", "--skip-cache-write", help="Skip cache write", action="store_true"
)
parser.add_argument("--verbose", "-v", action="count", default=0)
args = parser.parse_args()

logging.basicConfig()
logger = logging.getLogger("poorboy")

if args.verbose == 1:
    logger.setLevel(logging.INFO)
elif args.verbose > 1:
    logger.setLevel(logging.DEBUG)

result = Poorboy().calculate_security_buys(
    args.anchor_ticker,
    float(args.total_invest_available),
    force_refresh_cache=args.force_refresh_cache,
    skip_cache_write=args.skip_cache_write,
)
print(result.to_csv(index=False))
