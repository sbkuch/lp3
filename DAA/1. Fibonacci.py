# Fibonacci with step count

# Recursive Fibonacci with step count
steps_recursive = 0  # global variable to count recursive calls

def recursive_fibonacci(n):
    global steps_recursive
    steps_recursive += 1
    if n <= 1:
        return n
    else:
        return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)


# Non-recursive Fibonacci with step count
def non_recursive_fibonacci(n):
    steps_non_recursive = 0
    first = 0
    second = 1
    print(first)
    print(second)
    for i in range(2, n):
        steps_non_recursive += 1
        third = first + second
        first = second
        second = third
        print(third)
    print("Non-recursive step count:", steps_non_recursive)


if __name__ == "__main__":
    n = 10
    print("Recursive Fibonacci:")
    for i in range(n):
        print(recursive_fibonacci(i))
    print("Recursive step count:", steps_recursive)

    print("\nNon-Recursive Fibonacci:")
    non_recursive_fibonacci(n)
