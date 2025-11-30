from data_loader import get_top_50_us_stocks, get_price_history

symbols = get_top_50_us_stocks()
print("Universe size:", len(symbols))
print("First 10 symbols:", symbols[:10])

prices = get_price_history(symbols)

print("\nPrice DataFrame shape:", prices.shape)
print("\nLast 5 rows of prices:\n")
print(prices.tail())
