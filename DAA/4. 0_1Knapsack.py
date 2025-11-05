def solve_knapsack():
    val = [50, 100, 150, 200]   # values of items
    wt = [8, 16, 32, 40]        # weights of items
    W = 64                      # maximum capacity
    n = len(val)                # number of items

    # DP table where dp[i][w] = max value with first i items and capacity w
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Build table in bottom-up manner
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if wt[i - 1] <= w:
                dp[i][w] = max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    print("Maximum value in Knapsack =", dp[n][W])


if __name__ == "__main__":
    solve_knapsack()
