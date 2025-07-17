def backtest(prices, signals, initial=100000):
    cash = initial
    portfolio = 0
    for i in range(1, len(prices)):
        if signals[i-1] == 1:
            portfolio = cash / prices[i-1]
            cash = 0
        elif signals[i-1] == -1:
            cash = portfolio * prices[i-1]
            portfolio = 0
    total = cash + portfolio * prices[-1]
    return total