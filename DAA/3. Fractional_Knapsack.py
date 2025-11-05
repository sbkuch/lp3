def fractional_knapsack():
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    total_value = 0.0

    # Pair each item as (value/weight ratio, weight, value)
    items = sorted(zip(weights, values), key=lambda x: x[1] / x[0], reverse=True)

    for w, v in items:
        if capacity == 0:
            break
        if w <= capacity:
            # Take the whole item
            total_value += v
            capacity -= w
        else:
            # Take the fractional part
            fraction = capacity / w
            total_value += v * fraction
            capacity = 0  # bag is full

    print("Maximum value in Knapsack =", total_value)


if __name__ == "__main__":
    fractional_knapsack()
