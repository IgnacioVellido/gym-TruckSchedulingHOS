# Network
import numpy as np
import osmnx as ox
import networkx as nx

from datetime import time

def query_graph(place: str = 'Spain', 
                filter: str = '["highway"~"motorway|motorway_link|trunk|trunk_link"]'):
    ox.config(use_cache=True, log_console=True)

    G = ox.graph_from_place(place, network_type='drive',
                            simplify=True, custom_filter=filter)

    # Remove attributes

    # Simplify?
    # https://osmnx.readthedocs.io/en/stable/osmnx.html#module-osmnx.simplification

    # Plot graph
    ox.plot_graph(G)

    # Save graph
    # ox.save_graphml(G, 'spain-raw.graphml')

    return G


def remove_attributes(G):
    # DON'T EXECUTE WHILE TESTING, ONLY TO GET FINAL GRAPH

    # Edge attributes to remove
    att_list = [
        "maxspeed",
        'from',
        'to',
        'osmid',
        'ref',

        'width',
        "junction",
        "access",
        "tunnel",
        "lanes",
        "geometry",
        "bridge",
        "highway",
        "name",
        "oneway"
    ]

    for n1, n2, d in G.edges(data=True):
        for att in att_list:
            d.pop(att, None)

    # Node attributes to remove
    att_list = [
        "ref",

        "highway",
        "street_count",
        "x",  # Don't remove in RL
        "y"   # Don't remove in RL
    ]

    for n, d in G.nodes(data=True):
        for att in att_list:
            d.pop(att, None)

    return G


def time_to_min(t):
  """ Number of minutes in time object """
  return t.hour * 60 + t.minute


def preprocess_graph(G):
    G_nx = nx.relabel.convert_node_labels_to_integers(G)
    nodes, edges = ox.graph_to_gdfs(G_nx, nodes=True, edges=True)

    # Set node attributes (type and time windows)
    G_nx = set_node_attributes(G_nx)

    # Remove nodes with no output road
    G_nx = remove_deadends(G_nx)

    # Add speed to edges
    G_nx = set_graph_speed(G_nx)

    return G_nx


def set_node_attributes(G):
    # Set default values
    nx.set_node_attributes(G, 0, 'type') # Using numerical categories instead

    nx.set_node_attributes(G, time_to_min(time(0,0)), 'open_time')
    nx.set_node_attributes(G, time_to_min(time(23,59)), 'close_time')


    # Add random parking and stations
    num_graph_nodes = G.number_of_nodes()

    num_parking = int(num_graph_nodes * 0.1)
    num_stations = int(num_graph_nodes * 0.2)


    parking_nodes = np.random.choice(num_graph_nodes, num_parking, replace=False)
    for n in parking_nodes:
        G.nodes[n]['type'] = 1 # Using numerical categories instead

    # TODO: random open_close times
    G.nodes[n]['open_time'] = time_to_min(time(2,0))
    G.nodes[n]['close_time'] = time_to_min(time(23,0))

    # TODO: let stations be in some cases the same as parking - New type: station&parking
    station_nodes = np.random.choice(num_graph_nodes, num_stations, replace=False) 
    for n in station_nodes:
        # If already parking
        if G.nodes[n]['type'] == 1:
            G.nodes[n]['type'] = 3 # Parking & Station
        else:
            G.nodes[n]['type'] = 2 # Only parking

    # TODO: random open_close times
    G.nodes[n]['open_time'] = time_to_min(time(9,0))
    G.nodes[n]['close_time'] = time_to_min(time(23,0))

    return G

# ------------------------------------------------------------------------------

def remove_deadends(G):
    # Iterate until empty
    no_out = list(node for node, out_degree in G.out_degree() if out_degree == 0)
    while no_out:
      G.remove_nodes_from(no_out)
      G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')

      no_out = list(node for node, out_degree in G.out_degree() if out_degree == 0)

    return G

# ------------------------------------------------------------------------------

def set_graph_speed(G):
    # NOTE: maxspeed not available in all nodes

    speeds = [50000, 80000, 90000, 100000] 
    for u,v,att in G.edges(data=True):
        # TODO: Set maxspeed if defined
        # Sometimes maxspeed is str, other list(str)
        # if 'maxspeed' in att:
        #   if len(att['maxspeed'])
        #   print(att['maxspeed'])
        #   att['speed'] = int(att['maxspeed'][0]) * 1000
        #   print(att['speed'])
        # else:
        att['speed'] = np.random.choice(speeds, p=[0.1,0.4,0.4,0.1]) # Add random speed

    return G