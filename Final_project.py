import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd

file_path =
entities_df = pd.read_csv('file_path')
relationships_df = pd.read_csv('file_path')

combined_df = pd.merge(entities_df, relationships_df, left_on='node_id', right_on='node_id_start', how='inner')
print(combined_df)
variables = combined_df.columns
print(variables)

panama_df = combined_df[combined_df['sourceID_x'] == "Panama Papers"]
paradise_df = combined_df[combined_df['sourceID_x'].str.contains('Paradise Papers', case=False, na=False)]

countries_of_interest = ["Netherlands", "Luxembourg", "United Kingdom", "Switzerland", "Ireland"]

#["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
#"Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary"
#"Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
#"Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]

filtered_df_panama = panama_df[panama_df['countries'].isin(countries_of_interest)]

G_panama_1 = nx.Graph()

# Add nodes
nodes = filtered_df_panama["node_id_start"].unique()
G_panama_1.add_nodes_from(nodes)

# Add edges representing relations between entities
edges = filtered_df_panama[["node_id_start", "node_id_end"]].values.tolist()
G_panama_1.add_edges_from(edges)


# Plot the network
pos = nx.spring_layout(G_panama_1, seed=42)  # Seed for reproducibility
plt.figure(figsize=(12, 8))
nx.draw(G_panama_1, pos, with_labels=False, node_size=2, node_color="lightblue", font_size=6) #labels off for better view of the network
nx.draw_networkx_edges(G_panama_1, pos, alpha=1, width=1.0, edge_color='black', style='solid')
plt.title("Panama Papers Network in the Netherlands, UK, Ireland, Switzerland and Luxembourg")
plt.savefig("file_path/panama_1.png", format="png", dpi=300, bbox_inches='tight')
plt.show()

#straight from the SNA lecture, it's a little bit uninteresting as it just has a lot of lists and numbers
print(f"Nodes: {G_panama_1.nodes}") #bit useless because it just provides a very long list
print(f"Edges: {G_panama_1.edges}") #idem
print(f"Number of nodes: {G_panama_1.number_of_nodes()}")
print(f"Number of edges: {G_panama_1.number_of_edges()}")
print(f"Is the graph directed?: {G_panama_1.is_directed()}")
print(f"Network density: {nx.density(G_panama_1)}")
print(f"Node degrees: {G_panama_1.degree()}")
print(f"Number of components: {len(list(nx.connected_components(G_panama_1)))}")
print(f"Components: {list(nx.connected_components(G_panama_1))}")
#a couple of interesting things to note here: number of nodes: 514, number of edges: 350, undirected graph, network density: 0.002654712, number of components: 164
#interestingly, this is, of course for the country filtered data, if we take the unfiltered dat then it coms out as follows: nodes: 12283; edges: 8063; undirected graph;
#network density: 0.00010689394629077012; number of componenets: 4226. --> I think many of these numbers are still from the EU 27

#Establish a threshold for the minimum component size, as is seen abvoe there are quite some components
# Create the graph
G_panama_2 = nx.Graph()

# Add nodes
nodes = filtered_df_panama["node_id_start"].unique()
G_panama_2.add_nodes_from(nodes)

# Add edges representing relations between entities
edges = filtered_df_panama[["node_id_start", "node_id_end"]].values.tolist()
G_panama_2.add_edges_from(edges)

# Define a size threshold
size_threshold = 3  # Set the desired threshold here

# Find all connected components
connected_components = list(nx.connected_components(G_panama_2))

# Filter components larger than the threshold
large_components = [c for c in connected_components if len(c) >= size_threshold]

# Combine all large components into a new graph
G_filtered = G_panama_2.subgraph(set.union(*large_components)).copy()

# Plot the filtered graph
pos = nx.spring_layout(G_filtered, seed=42)  # Seed for reproducibility
plt.figure(figsize=(12, 8))
nx.draw(G_filtered, pos, with_labels=False, node_size=2, node_color="lightblue", font_size=6)
nx.draw_networkx_edges(G_filtered, pos, alpha=1, width=1.0, edge_color='black', style='solid')
plt.title("Filtered Panama Papers Network in the Netherlands, UK, Ireland, Switzerland and Luxembourg")
plt.savefig("file_path/panama_2.png", format="png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Nodes: {G_filtered.nodes}") #bit useless because it just provides a very long list
print(f"Edges: {G_filtered.edges}") #idem
print(f"Number of nodes: {G_filtered.number_of_nodes()}")
print(f"Number of edges: {G_filtered.number_of_edges()}")
print(f"Is the graph directed?: {G_filtered.is_directed()}")
print(f"Network density: {nx.density(G_filtered)}")
print(f"Node degrees: {G_filtered.degree()}")
print(f"Number of components: {len(list(nx.connected_components(G_filtered)))}")
print(f"Components: {list(nx.connected_components(G_filtered))}")

#Network centrality for both graphs
#Panama 1
degree_centrality_panama_1= nx.degree_centrality(G_panama_1)
top_nodes = sorted(degree_centrality_panama_1, key=degree_centrality_panama_1.get, reverse=True)[:5]

# Print the top five nodes with their degree centrality values
print("Top 5 nodes with highest degree centrality:")
for node in top_nodes:
    centrality = degree_centrality_panama_1[node]
    print(f"Node {node}: Degree Centrality {centrality}")

# Visualization
pos = nx.spring_layout(G_panama_1, seed=42)  # Seed for reproducibility
plt.figure(figsize=(12, 8))

# Draw nodes with sizes proportional to their degree centrality
node_sizes = [5000 * degree_centrality_panama_1[node] for node in G_panama_1.nodes()]
nx.draw(G_panama_1, pos, with_labels=False, node_size=node_sizes, node_color="lightblue", font_size=6)
nx.draw_networkx_edges(G_panama_1, pos, alpha=1, width=1.0, edge_color='black', style='solid')

# Highlight the top five nodes with the highest degree centrality
nx.draw_networkx_nodes(G_panama_1, pos, nodelist=top_nodes, node_size=250, node_color="red")

plt.title("Panama Papers Network with Node Sizes Proportional to Degree Centrality")
plt.savefig("file_path/panama_centrality_1.png", format="png", dpi=300, bbox_inches='tight')
plt.show()

#Panama 2
degree_centrality_panama_2= nx.degree_centrality(G_panama_2)
top_nodes = sorted(degree_centrality_panama_2, key=degree_centrality_panama_2.get, reverse=True)[:5]

# Print the top five nodes with their degree centrality values
print("Top 5 nodes with highest degree centrality:")
for node in top_nodes:
    centrality = degree_centrality_panama_2[node]
    print(f"Node {node}: Degree Centrality {centrality}")

# Visualization
pos = nx.spring_layout(G_panama_2, seed=42)  # Seed for reproducibility
plt.figure(figsize=(12, 8))

# Draw nodes with sizes proportional to their degree centrality
node_sizes = [5000 * degree_centrality_panama_2[node] for node in G_panama_2.nodes()]
nx.draw(G_panama_2, pos, with_labels=False, node_size=node_sizes, node_color="lightblue", font_size=6)
nx.draw_networkx_edges(G_panama_2, pos, alpha=1, width=1.0, edge_color='black', style='solid')

# Highlight the top five nodes with the highest degree centrality
nx.draw_networkx_nodes(G_panama_2, pos, nodelist=top_nodes, node_size=250, node_color="red")
plt.title("Panama Papers Network with Node Sizes Proportional to Degree Centrality (Minimum component size of 3)")
plt.savefig("file_path/panama_centrality_2.png", format="png", dpi=300, bbox_inches='tight')

plt.show()


####################========================================####################
####################==========PARADISE PAPERS SNA===========####################
####################========================================####################
#Now doing all of the same as the above but for the Paradise papers.
filtered_df_paradise = paradise_df[paradise_df['countries'].isin(countries_of_interest)]

G_paradise_1 = nx.Graph()

# Add nodes
nodes = filtered_df_paradise["node_id_start"].unique()
G_paradise_1.add_nodes_from(nodes)

# Add edges representing relations between entities
edges = filtered_df_paradise[["node_id_start", "node_id_end"]].values.tolist()
G_paradise_1.add_edges_from(edges)

# Plot the network
pos = nx.spring_layout(G_paradise_1, seed=42)  # Seed for reproducibility
plt.figure(figsize=(12, 8))
nx.draw(G_paradise_1, pos, with_labels=False, node_size=5, node_color="lightblue", font_size=8)
nx.draw_networkx_edges(G_paradise_1, pos, alpha=1, width=1.0, edge_color='black', style='solid')
plt.title("Paradise Papers Network in the Netherlands, UK, Ireland, Switzerland and Luxembourg")
plt.savefig("file_path/paradise_1.png", format="png", dpi=300, bbox_inches='tight')

plt.show()

#straight from the SNA lecture, it's a little bit uninteresting as it just has a lot of lists and numbers
print(f"Nodes: {G_paradise_1.nodes}") #bit useless because it just provides a very long list
print(f"Edges: {G_paradise_1.edges}") #idem
print(f"Number of nodes: {G_paradise_1.number_of_nodes()}")
print(f"Number of edges: {G_paradise_1.number_of_edges()}")
print(f"Is the graph directed?: {G_paradise_1.is_directed()}")
print(f"Network density: {nx.density(G_paradise_1)}")
print(f"Node degrees: {G_paradise_1.degree()}")
print(f"Number of components: {len(list(nx.connected_components(G_paradise_1)))}")
print(f"Components: {list(nx.connected_components(G_paradise_1))}")


G_paradise_2 = nx.Graph()
# Add nodes
nodes = filtered_df_paradise["node_id_start"].unique()
G_paradise_2.add_nodes_from(nodes)

# Add edges representing relations between entities
edges = filtered_df_paradise[["node_id_start", "node_id_end"]].values.tolist()
G_paradise_2.add_edges_from(edges)

# Define a size threshold
size_threshold = 3  # Set the desired threshold here

# Find all connected components
connected_components = list(nx.connected_components(G_paradise_2))

# Filter components larger than the threshold
large_components = [c for c in connected_components if len(c) >= size_threshold]

# Combine all large components into a new graph
G_filtered_paradise = G_paradise_2.subgraph(set.union(*large_components)).copy()

# Plot the filtered graph
pos = nx.spring_layout(G_filtered_paradise, seed=42)  # Seed for reproducibility
plt.figure(figsize=(12, 8))
nx.draw(G_filtered_paradise, pos, with_labels=False, node_size=5, node_color="lightblue", font_size=6)
nx.draw_networkx_edges(G_filtered_paradise, pos, alpha=1, width=1.0, edge_color='black', style='solid')
plt.title("Filtered Paradise Papers Network in the Netherlands, UK, Ireland, Switzerland and Luxembourg")
plt.savefig("file_path/paradise_2.png", format="png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Nodes: {G_filtered_paradise.nodes}") #bit useless because it just provides a very long list
print(f"Edges: {G_filtered_paradise.edges}") #idem
print(f"Number of nodes: {G_filtered_paradise.number_of_nodes()}")
print(f"Number of edges: {G_filtered_paradise.number_of_edges()}")
print(f"Is the graph directed?: {G_filtered_paradise.is_directed()}")
print(f"Network density: {nx.density(G_filtered_paradise)}")
print(f"Node degrees: {G_filtered_paradise.degree()}")
print(f"Number of components: {len(list(nx.connected_components(G_filtered_paradise)))}")
print(f"Components: {list(nx.connected_components(G_filtered_paradise))}")


#Degree centrality for the paradise papers
#Paradise 1
degree_centrality_paradise_1= nx.degree_centrality(G_paradise_1)
top_nodes = sorted(degree_centrality_paradise_1, key=degree_centrality_paradise_1.get, reverse=True)[:5]

# Print the top five nodes with their degree centrality values
print("Top 5 nodes with highest degree centrality:")
for node in top_nodes:
    centrality = degree_centrality_paradise_1[node]
    print(f"Node {node}: Degree Centrality {centrality}")

# Visualization
pos = nx.spring_layout(G_paradise_1, seed=42)  # Seed for reproducibility
plt.figure(figsize=(12, 8))

# Draw nodes with sizes proportional to their degree centrality
node_sizes = [5000 * degree_centrality_paradise_1[node] for node in G_paradise_1.nodes()]
nx.draw(G_paradise_1, pos, with_labels=False, node_size=node_sizes, node_color="lightblue", font_size=6)
nx.draw_networkx_edges(G_paradise_1, pos, alpha=1, width=1.0, edge_color='black', style='solid')

# Highlight the top five nodes with the highest degree centrality
nx.draw_networkx_nodes(G_paradise_1, pos, nodelist=top_nodes, node_size=100, node_color="red")
plt.title("Paradise Papers Network with Node Sizes Proportional to Degree Centrality")
plt.savefig("file_path/paradise_centrality_1.png", format="png", dpi=300, bbox_inches='tight')
plt.show()

#Paradise 2
degree_centrality_paradise_2= nx.degree_centrality(G_paradise_2)
top_nodes = sorted(degree_centrality_paradise_2, key=degree_centrality_paradise_2.get, reverse=True)[:5]

# Print the top five nodes with their degree centrality values
print("Top 5 nodes with highest degree centrality:")
for node in top_nodes:
    centrality = degree_centrality_paradise_2[node]
    print(f"Node {node}: Degree Centrality {centrality}")

# Visualization
pos = nx.spring_layout(G_paradise_2, seed=42)  # Seed for reproducibility
plt.figure(figsize=(12, 8))

# Draw nodes with sizes proportional to their degree centrality
node_sizes = [5000 * degree_centrality_paradise_2[node] for node in G_paradise_2.nodes()]
nx.draw(G_paradise_2, pos, with_labels=False, node_size=node_sizes, node_color="lightblue", font_size=6)
nx.draw_networkx_edges(G_paradise_2, pos, alpha=1, width=1.0, edge_color='black', style='solid')

# Highlight the top five nodes with the highest degree centrality
nx.draw_networkx_nodes(G_paradise_2, pos, nodelist=top_nodes, node_size=100, node_color="red")
plt.title("Paradise Papers Network with Node Sizes Proportional to Degree Centrality")
plt.savefig("file_path/paradise_centrality_2.png", format="png", dpi=300, bbox_inches='tight')
plt.show()

#We have the largest nodes for each network now it's time to find out what country each of these correpsonds to,
#this is only done for the centrality graphs without the threshold. This is because including a threshold for the minimum size of a 
#component likely would not change the outcome, the minimum threshold after all only removes those components that have low amounts of connection
#here I'm only looking at the nodes with the highest amounts of connections (edges).

dc_df_panama_1 =  pd.DataFrame(degree_centrality_panama_1.items(), columns=['node_id_end', 'degree_centrality'])
print(dc_df_panama_1)

#only 5 biggest nodes
top_n = 5
# Sort the DataFrame by degree centrality in descending order and get the top n nodes
top_nodes = dc_df_panama_1.nlargest(top_n, 'degree_centrality')
# Extract the node_ids of the top n nodes
top_node_ids = top_nodes['node_id_end'].tolist()
# Display the top nodes
print(top_nodes)

#match the node_id to the row in the original CSV, this way it is see what country those entities (nodes) are in.
matching_rows_panama_1 = combined_df[combined_df['node_id_end'].isin(top_node_ids)]
print(matching_rows_panama_1)
matching_rows_panama_1 = matching_rows_panama_1.head(top_n)
country_names = matching_rows_panama_1['countries'].tolist()
print("Country Names Corresponding to Top Nodes:")
print(country_names)


#Do the same as the above to find the most important countries in the paradise papers.

dc_df_paradise_1 =  pd.DataFrame(degree_centrality_paradise_1.items(), columns=['node_id_end', 'degree_centrality'])
print(dc_df_paradise_1)

#only 5 biggest nodes
top_n = 5
# Sort the DataFrame by degree centrality in descending order and get the top n nodes
top_nodes = dc_df_paradise_1.nlargest(top_n, 'degree_centrality')
# Extract the node_ids of the top n nodes
top_node_ids = top_nodes['node_id_end'].tolist()
# Display the top nodes
print(top_nodes)

#match the node_id to the row in the original CSV, this way it is see what country those entities (nodes) are in.
matching_rows_paradise_1 = combined_df[combined_df['node_id_end'].isin(top_node_ids)]
print(matching_rows_paradise_1)
matching_rows_paradise_1 = matching_rows_paradise_1.head(top_n)
country_names = matching_rows_paradise_1['countries'].tolist()
print("Country Names Corresponding to Top Nodes:")
print(country_names)