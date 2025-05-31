from polygon import RESTClient

client = RESTClient("w2chIPf4EUplQqQv6b8Nxnn8GQV8pfGC")

aggs = []
for a in client.list_aggs(
    "O:SPY251219C00650000",
    1,
    "day",
    "2023-01-09",
    "2023-02-10",
    adjusted="true",
    sort="asc",
    limit=120,
):
    aggs.append(a)

print(aggs)