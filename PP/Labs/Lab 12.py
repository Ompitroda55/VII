import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()

def display_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12)
    plt.title("Network Graph")
    plt.axis('off')
    plt.show()

G.add_node(1)
display_graph(G)  # Display the graph after adding a node

G.add_nodes_from([2, 3, 4, 7, 9])
G.add_edges_from([(1, 2), (3, 1), (2, 4), (4, 1), (9, 1), (1, 7), (2, 9)])
display_graph(G)  # Display the graph after adding nodes and edges

G.remove_node(3)
display_graph(G)  # Display the graph after removing a node

G.remove_edge(1, 2)
display_graph(G)  # Display the graph after removing an edge

n = G.number_of_nodes()
m = G.number_of_edges()
print("Number of nodes:", n)
print("Number of edges:", m)

d = G.degree(2)
print("Degree of node 2:", d)

neighbor_list = list(G.neighbors(2))
print("Neighbors of node 2:", neighbor_list)

G.clear()


data = [25, 30, 15, 10, 20]
labels = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']

plt.figure(1)
plt.pie(data, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart')

plt.figure(2)
plt.bar(labels, data)
plt.title('Bar Chart')

x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.figure(3)
plt.plot(x, y)
plt.title('Line Plot')

x = np.random.rand(50)
y = np.random.rand(50)
plt.figure(4)
plt.scatter(x, y)
plt.title('Scatter Plot')

data = np.random.normal(0, 1, 1000)
plt.figure(5)
plt.hist(data, bins=30)
plt.title('Histogram')

plt.show()

