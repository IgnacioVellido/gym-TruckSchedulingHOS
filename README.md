# gym-TruckSchedulingHOS

A gym environment to solve the Truck Scheduling Problem with hours of service. Supports additional metrics like refueling, safe parking and time windows.

## Problem representation

- Input:
  - Road map as a graph (static).
    - Nodes: For efficiency it should only contain non-normal nodes and intersections
      - type:
        - normal
        - parking
        - station
        - parking&station
      - open_time: Start time of time window (minutes, 0=00:01)
      - close_time: Closing time of time window (minutes, 0=00:01)
    - Edges: Multiple edges between two nodes allowed only if they go in opposite directions
      - length: meters between both nodes
      - speed: Average speed (meters/hour) during the edge

  - Status (changes at each step)
    - current_node (int): Index of agents node
    - target_node (int): Index of target node

    - seq_t (int):  Accumulated HoS in actual sequence (minutes)
    - day_t (int):  Accumulated HoS in actual day (minutes)
    - week_t (int): Accumulated HoS in actual week (minutes)

    - fuel (int): Actual fuel in tank
    - max_fuel (int): Max fuel in tank

- Actions [#nodes + 4]
  - Select edge = next connected node to transverse
  - Refuel tank iff current_node.type=station
  - Rests at node iff current_node.type= (seq, day or week)

- Solution: List of actions, from starting node to target node. With constraints:
  - Reaching client out of time windows     (Soft)
  - Committing HoS infraction      (Soft)
    - Missing seq rest
    - Missing day rest
    - Missing week rest
  - Moving to non-adjacent node     (Hard)
  - Emptying fuel tank at any moment    (Hard)

- Reward/Quality/Cost
  - Note: Regret values can be modified
  - Natural (informative) (at each step): Negative, maximization problem
    - \- (distance (km) + duration (h) + fuel (L) + infractions_penalty)

        ```{}
        infractions_penalty = tw_infractions + hos_infractions + fuel_infractions

        tw_infractions = REGRET_TW + hours from time window   if out of time window
                         0   otherwise
            REGRET_TW = 100

        fuel_infractions = REGRET_FUEL if actual_fuel < 0
                           0   otherwise
            REGRET_FUEL = 999

        hos_infraction  = 0
        hos_infraction += REGRET_SEQT if seq_t  > MAX_SEQ_T  else 0
        hos_infraction += REGRET_DAYT if day_t  > MAX_DAY_T  else 0
        hos_infraction += REGRET_WEET if week_t > MAX_WEEK_T else 0

            REGRET_SEQT = 50
            REGRET_DAYT = 100
            REGRET_WEET = 200
        ```

  - Shaped reward (RL agent will take a lot of time to update, NOT RECOMMENDED):
    - (length + duration + tw + fuel + HoS)    in client node

    ```{}
        - total_distance = 1 for each kilometer
        - total_duration = 1 for each minute
        - total_fuel     = 1 for each liter
        - total_tw_infractions   = Sum of (0 if arriving in time_window) (else 5 + 1 for each minute of difference) at each node
        - total_hos_infractions  = Sum of hos infractions at each node
        - total_fuel_infractions = Sum of fuel infractions at each node
    - zero otherwise
    ```

## Notes

- Data examples include different versions of the Andalusia and Spain main roads map. You can use the script ```data/osm-extractor.py``` to extract your own graphs.

- Some architectures (Double-Deep-Q Network, Message Passing Network) are included on the ```models``` subfolder, although they have not been used to solve the problem.

- Conda environment provided in ```requirements.txt```
