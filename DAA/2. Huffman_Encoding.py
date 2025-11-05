import heapq

# Create Huffman tree node
class node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq           # Frequency of the symbol
        self.symbol = symbol       # Symbol name (character)
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.huff = ''             # Huffman direction (0 or 1)

    def __lt__(self, nxt):
        # To compare two nodes based on frequency (for heap)
        return self.freq < nxt.freq

# Function to print Huffman codes
def printnodes(node, val=''):
    newval = val + str(node.huff)
    # If node has left child, traverse left
    if node.left:
        printnodes(node.left, newval)
    # If node has right child, traverse right
    if node.right:
        printnodes(node.right, newval)
    # If node is a leaf, print the symbol and its code
    if not node.left and not node.right:
        print(f"{node.symbol} -> {newval}")

if __name__ == "__main__":
    chars = ['a', 'b', 'c', 'd', 'e', 'f']
    freq = [5, 9, 12, 13, 16, 45]
    nodes = []

    # Step 1: Create initial heap
    for i in range(len(chars)):
        heapq.heappush(nodes, node(freq[i], chars[i]))

    # Step 2: Combine nodes until one remains
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)

        left.huff = 0
        right.huff = 1

        # Create new internal node with combined frequency
        newnode = node(left.freq + right.freq, left.symbol + right.symbol, left, right)
        heapq.heappush(nodes, newnode)

    # Step 3: Print Huffman codes
    print("Huffman Codes are:")
    printnodes(nodes[0])
