from data_loader import get_top_50_us_stocks, get_price_history
from features_karina import compute_returns, estimate_mu_sigma


symbols = get_top_50_us_stocks()
prices = get_price_history(symbols)

returns = compute_returns(prices)
mu, sigma = estimate_mu_sigma(returns)

print("Returns shape:", returns.shape)
print("Mu shape:", mu.shape)
print("Sigma shape:", sigma.shape)

print("\nFirst 5 expected returns:")
print(mu[:5])

print("\nCovariance matrix top-left 5x5:")
print(sigma[:5, :5])
