# TruckScheduling-GNN

Solving the truck scheduling problem with HoS. Plus refueling, safe parking and time windows.

## Problem representation

- Input:
  - Road map as a graph (static).
    - Nodes: It **should** (not what has been tested with) only contain non-normal nodes and intersections
      - type:
        - normal
        - parking
        - station
        - parking&station
        - (if VRP) depot
      - open_time: Start time of time window (minutes, 0=00:01)
      - close_time: Closing time of time window (minutes, 0=00:01)
    - Edges: Multiple edges between two nodes allowed only if they go in opposite directions
      - length: meters between both nodes
      - speed: Average speed (meters/hour) during the edge

  - Status (variable at each step)
    - current_node (int): Index of agents node
    - target_node (int): Index of target node

    - seq_t (int):  Accumulated HoS in actual sequence (minutes)
    - day_t (int):  Accumulated HoS in actual day (minutes)
    - week_t (int): Accumulated HoS in actual week (minutes)

    - fuel (int): Actual fuel in tank
    - max_fuel (int): Max fuel in tank

    - (if vrp) cargo

- Actions [#nodes + 4]
  - Select edge = next connected node to transverse
  - Refuel tank if station
  - Rest at node if parking (seq, day or week)
  - (not used) Split and reduced rests
  - (not used) (if vrp) load/unload cargo

- Solution: List of actions, from starting node to target node. With constraints:
  - Reaching client out of time windows     (Soft)
  - Commiting HoS infraction      (Soft)
    - Missing seq rest
    - Missing day rest
    - Missing week rest
  - Moving to non-adjacent node     (Hard)
  - Emptying fuel tank at any moment    (Hard)
  - (If VRP) move to client with no cargo on truck (Hard)

- Reward/Quality/Cost
  - Natural (at each step): Negative, maximization problem
    - \- (distance (km) + duration (h) + fuel (L) + infractions_penalty)

        ```{}
        infractions_penalty = tw_infractions + hos_infractions + fuel_infractions

        tw_infractions = 100 + hours from time window   if out of time window
                                                    0   otherwise

        fuel_infractions = 999   if actual_fuel < 0
                             0   otherwise

        hos_infraction  = 0
        hos_infraction +=  50 if seq_t  > MAX_SEQ_T  else 0
        hos_infraction += 100 if day_t  > MAX_DAY_T  else 0
        hos_infraction += 200 if week_t > MAX_WEEK_T else 0
        ```

## Possible architectures

Things to consider:

- Techniques:
  - Using normal planning, or another method, as a baseline

  - Using GNN to predict Q value of a DQN

  - Using GNN (GAT, residual E-GAT maybe) to predict node, use Linear Layer to get embedding of current status. Concat with Linear (+ mask) and predict best-action

  - Need to generalize over multiple graphs for scalability. If restricting at only one graph, needs to generalize over multiple path.

  - If there is no information about distance to each node will be difficult to generalize, as the model does not know with path take torwards the target

  - RL agent will need lots of iterations and will end learning best paths for pair of nodes, but probably won't generalize to new graphs (no information on how to reach the target from a first visited node)

  - Nevertheless, if it works in great scale graphs (combined with other ideas, like planning), a company could train it for many iterations and just deploy it always in the same environment

  - Using an ensemble with other prediction models (cost, delay...) for the cost/reward function

  - Optimal solution could not be found, so an improvement local search process could be included (advantegeous because inference is fast and there is no need for real time planning)

- Representation:
    1. Add self-loops to each parking/station node, each loop would indicate one of the extra actions (with distance being the time of rest/refuel). That way, the GNN could predict the path without RL

    2. Separate parking and station as additional nodes between edges, where the agent can choose to go through there or not. Travelling to that node indicates doing the action Rest/refuel. Maybe the distance indicates the time spent doing the action, and additional edge features could be included to indicate the refueling and the recharge or HoS (like, when moving to the node the fuel increases +3000 clipped to the max, and HoS is set to max...).

       But how to treat time windows at those nodes? The computation is not easy as just passing the graph to the network, a mask that depends of the actual time should be used AFTER (or maybe before setting to 0)

    3. Instead of a huge size input path, use planning to get multiple paths, then expand each node with a few neighbors, and use the combination of all graphs as input

    4. Training with a big sized map, using portions of the map at inference (for example, with me method proposed above)

    5. Hierarchical/nested graphs, predict rests in simplified graphs and plan in more detailed ones

    6. Our graphs have few connections between them. Virtual edges could be added to increse speed of message passing through the network (**needs small input graph**)

### 1. Fixed GNN + RL

Because graph information is static, train a message passing GNN with only the graph. Using it at each RL step as inference to use in a RL agent

### 2. Recursively trained GNN + RL

Train the RL agent and a GNN at the same time, but using the previous iteration graph as the current graph

### ? 3. GNN + Planning

Train a GNN to get a heuristic value for each node, using planning with that heuristic.

**DOUBT**: How the temporal status is considered, maybe a temporal dimension for each node?

### ? 4. RL + MonteCarlo Tree search

??

### ? 5. Only GNN

Let GNN predict path. See Representation point 1.

**DOUBT**: How the temporal status is considered, maybe a temporal dimension for each node?

**DOUBT**: The output is not fixed, is stochastic

### ? 6. GNN as Q value predictor

Inspired by (read more carefully): <https://arxiv.org/pdf/1910.07421.pdf>

Given the network with the status as global features, predict Q value for each action (using mask)

## Related papers

### Reinforcement Learning for Solving the Vehicle Routing Problem

<https://proceedings.neurips.cc/paper/2018/hash/9fb4651c05b2ed70fba5afe0b039a550-Abstract.html>

VRP. RNN + attention. Policy gradient + actor-critic. Returns one action each step.

### Deep Reinforcement Learning for the Electric Vehicle Routing Problem With Time Windows

<https://ieeexplore.ieee.org/document/9520134?arnumber=9520134>

EVRPTW. GNNCONV + structure2vec + attention + LSTM. Policy gradient + REINFORCE. Small experiments (small computation time)

Going to station node means charging

### SOLO: Search Online, Learn Offline for Combinatorial Optimization Problems

<https://arxiv.org/abs/2104.01646>

CVRP + PMSP. MPN GNN + embedding. DQN for heuristic, MCTS (UCT) for planning using heuristic.

### * Online Vehicle Routing With Neural Combinatorial Optimization and Deep Reinforcement Learning

<https://cse.sustech.edu.cn/faculty/~james/files/publications/Online-Vehicle-Routing-with-Neural-Combinatorial-Optimization-and-Deep-Reinforcement-Learning.pdf>

### Alphazero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

<https://arxiv.org/abs/1712.01815>

### Other useful bibliography

- Deep Reinforcement Learning meets Graph Neural Networks: exploring a routing optimization use case
    <https://arxiv.org/pdf/1910.07421.pdf>

    Bandwith routing problem: GNN to predict Q value, action space embedded in DQN, with RNN

    ```the action should be introduced into the agent in the form of a graph.
    This makes the action representation invariant to node and edge
    permutation, which means that, once the GNN is successfully trained, it 
    should be able to understand actions over arbitrary graph structures (i.e., 
    over different network states and topologies)```

- Solve routing problems with a residual edge-graph attention neural network

    <https://arxiv.org/pdf/2105.02730.pdf>

    CVRP. Encoder-Decoder, attention, recursive, residual E-GAT. REINFORCE / PPO

- ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!

    <https://openreview.net/pdf?id=ByxBFsRqYm>

## Things to remember

- In RL, reward is better at each time step than at the end, because there is no vanishing problems and the agent can be learn frequently. On the other hand, a reward can be misleading if it is giving values when it shouldn't

## Paper possibilities

1. Environment (configurable, multiple rewards depending on tw, depot...), datasets (full region, smaller with more detail), problems (normal transport, torwards vrp...)
2. Solution
3. Solution
4. ...
