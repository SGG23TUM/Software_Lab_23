import json
import os
import networkx as nx
import matplotlib.pyplot as plt

# Define the mapping from predicates to their names
predicate_to_name = {
    "hanging from": "hanging from",
    "mounted on": "mounted on",
    "behind": "behind",
    "in front of": "in front of",
    "next to": "next to",
    "above": "above",
    "under": "under",
    "on": "on",
    "has opening in": "has opening in",
    "standing on": "standing on",
    "placed on": "placed on",
    "equipped with": "equipped with",
    "fixed on": "fixed on",
    "attached to": "attached to"
}

# Define the mapping from object IDs to their labels
idx_to_label = {
    1: "ceiling",
    2: "lighting",
    3: "speaker",
    4: "wall",
    5: "door",
    6: "smoke alarm",
    7: "floor",
    8: "trash bin",
    9: "elevator button",
    10: "escape sign",
    11: "board",
    12: "fire extinguisher",
    13: "door sign",
    14: "light switch",
    15: "emergency button",
    16: "elevator",
    17: "handrail",
    18: "show window",
    19: "pipes",
    20: "staircase",
    21: "window",
    22: "radiator",
    23: "stecker"
}

# Directory where JSON files are stored
directory = 'dataset/'

# Function to create and save graph from JSON
def create_graph_from_json(json_file):
    # Load JSON data
    with open(json_file) as file:
        data = json.load(file)

    G = nx.DiGraph()
    
    # Add nodes with labels
    for obj in data['objects']:
        node_label = idx_to_label.get(obj['object_id'], f"Unknown_{obj['object_id']}")
        G.add_node(node_label, **obj)

    # Add edges with labels, avoiding duplicates
    seen_triplets = set()
    for rel in data['relationships']:
        subj_label = idx_to_label.get(rel['subject_id'], f"Unknown_{rel['subject_id']}")
        obj_label = idx_to_label.get(rel['object_id'], f"Unknown_{rel['object_id']}")
        predicate_label = predicate_to_name.get(rel['predicate'], f"Unknown_{rel['predicate']}")
        
        triplet = (subj_label, obj_label, predicate_label)
        if triplet not in seen_triplets:
            G.add_edge(subj_label, obj_label, label=predicate_label)
            seen_triplets.add(triplet)

    pos = nx.spring_layout(G, k=0.3, iterations=20)  # with increasing -k- values, distance between nodes are increasing!!
    
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    # Draw the graph
    #pos = nx.spring_layout(G)
    #nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_size=8, font_weight='bold')
    #edge_labels = nx.get_edge_attributes(G, 'label')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    # Save the graph to a file
    plt.savefig(json_file.replace('.json', '.png'), format='PNG')
    plt.close()

# Process each JSON file
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        create_graph_from_json(os.path.join(directory, filename))
