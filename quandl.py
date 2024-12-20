import bitcoin.core
from bitcoin.rpc import RawProxy
from bitcoin.wallet import CBitcoinAddress

# Connect to a local Bitcoin node
proxy = RawProxy()
# Fetch the latest block
block_count = proxy.getblockcount()
block_hash = proxy.getblockhash(block_count)

# Get the transactions in the block
txs = proxy.getblock(block_hash)['tx']

# Initialize counters
long_term_holders = 0
short_term_holders = 0

# Define the threshold in days
threshold = 155 * 24 * 60 * 60 # 155 days in seconds
for tx in txs:
    for out in tx['vout']:
        if out['scriptPubKey']['type'] == 'nonstandard':
            continue

        addr = CBitcoinAddress.from_scriptPubKey(out['scriptPubKey'])
        balance = proxy.getreceivedbyaddress(addr)
        if balance > threshold:
            long_term_holders += 1
        else:
            short_term_holders += 1

print(f"Long-term holders: {long_term_holders}")
print(f"Short-term holders: {short_term_holders}")
