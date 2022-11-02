import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

# Network
import osmnx as ox
import networkx as nx

# RL environment
import gym
from gym import spaces

# for plotting
import matplotlib.pyplot as plt
import geopandas


class TruckRouting(gym.Env):
    """
    Environment for the truck scheduling problem
    with HoS management, parking, fuel consumption and time windows
    """

    # Constants

    # Node type mapping
    TYPE_NORMAL = 0
    TYPE_PARKING = 1
    TYPE_STATION = 2
    TYPE_PARKING_STATION = 3
    # TYPE_DEPOT = 4 # Here you can refuel and rest 24/7

    # Fuel consumption rate (L/m)
    FUEL_RATE = 0.0003

    # HoS maximum times without break (minutes)
    MAX_SEQ_T = 4.5*60
    MAX_DAY_T = 10*60
    MAX_WEEK_T = 50*60

    # Duration of actions
    REFUEL_TIME = 15 # min
    SEQ_REST    = 45 # min
    DAILY_REST  = 11 # hours
    WEEKLY_REST = 45 # hours

    # Weights of metrics used in reward
    REGRET_FUEL = 999
    REGRET_TW   = 100
    REGRET_SEQT = 50
    REGRET_DAYT = 100
    REGRET_WEET = 200

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def __init__(self, config):
        """
        Initializes a new environment
        config: {
        'network': Map road network as a a Networkx graph. All nodes/edges must
                    have these properties:
            - Nodes:
              - type: normal, client, parking, station
              - open_time: Start time of time window (minutes, 0=00:01)
              - close_time: Closing time of time window (minutes, 0=00:01)
            - Edges:
              - length: meters between both nodes
              - speed: Average speed (meters/hour) during the edge

        'max_fuel': Maximum capacity of fuel in tank
        }
        """
        super(TruckRouting, self).__init__()

        # Set environment configuration
        self._set_config(config)
        self.reset()

        # Observations are dictionaries with the driver's and the client's location
        # + the network graph (always the same)
        # + current and target node
        self.observation_space = spaces.Dict({
            "current_node": spaces.Discrete(self.number_of_nodes),
            "target_node": spaces.Discrete(self.number_of_nodes),

            "seq_t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),
            "day_t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),
            "week_t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),

            "fuel": spaces.Box(low=0, high=self.max_fuel, shape=(1,), dtype=np.float32),
            # "max_fuel": spaces.Discrete(1),

            "available_actions": spaces.Box(low=0, high=1,
                                            shape=(self.number_of_nodes+4,),
                                            dtype=np.int64)
        })

        # Set number of possible actions: Rests + Refuel + Nodes in network
        self.action_space = spaces.Discrete(self.number_of_nodes + 4)


        # Print information about the environment
        print(f"Created environment with a network of {self.number_of_nodes} nodes.")

    # --------------------------------------------------------------------------

    def _get_obs(self):
        """ 
        Construct an observation for the actual environment state:

        current_node (int): Index of agents node
        target_node (int): Index of target node
        
        seq_t (int):  Accumulated HoS in actual sequence (minutes)
        day_t (int):  Accumulated HoS in actual day (minutes)
        week_t (int): Accumulated HoS in actual week (minutes)

        fuel (int): Actual fuel in tank

        available_actions (np.array(int)): Boolean mask of available actions in current state
        """
        return {
            "current_node": int(self.current_node),
            "target_node": int(self.target_node),
            
            "seq_t": np.array([self.seq_t], dtype=np.int64),
            "day_t": np.array([self.day_t], dtype=np.int64),
            "week_t": np.array([self.week_t], dtype=np.int64),

            "fuel": np.array([self.fuel], dtype=np.float32),

            "available_actions": self.mask
        }

    def _get_info(self):
        """ 
        Returns additional info of the observation:

        network (networkx graph): Networkx graph of the environment

        current_time (datetime): Current datetime object
        deadline (datetime): Deadline datetime object

        max_fuel (int): Tank capacity - TODO: DOES THE AGENT REALLY NEED THIS? 
                    THE ACTION IS TO REFUEL NO MATTER THE QUANTITY
                    ONLY USEFULL IS THE QUANTITY IS SELECTED

                    OR MAYBE IT LEARNS THE DIFFERENCE BETWEEN FUEL
                    AND MAX_FUEL
        """
        return {
            "network": self.network,
            
            "current_time": self.current_time,
            "deadline": self.deadline,

            "max_fuel": self.max_fuel
        }

    # ----------------------------------------------------------------------------

    def _set_config(self, config):
        """ Sets the initial parameters of the environment """
        self.network      = config['network']
        
        # Needed when using OSMnx
        self.network = nx.relabel.convert_node_labels_to_integers(self.network,
                                                                first_label=0,
                                                                ordering='default')
        
        self.nodes = self.network.nodes()  # Needed to comfortably access node info

        # Fuel
        self.max_fuel = config['max_fuel']

        # Could be first dimension if network as adjacency matrix
        self.number_of_nodes = self.network.number_of_nodes() 

        # Constant action maps for clearer code
        self.REST_SEQ = self.number_of_nodes
        self.REST_DAY = self.number_of_nodes + 1
        self.REST_WEEK = self.number_of_nodes + 2   # TODO: Add split rests

        self.REFUEL = self.number_of_nodes + 3

        # For plotting
        self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.network, nodes=True, edges=True)
        color_map = {0: 'w', 1: 'c', 2: 'y', 3: 'g'}

        self.colors = self.gdf_nodes['type'].replace(color_map)

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------

    def _get_path_length(self):
        """ Returns the number of steps in shortest path and its distance in km """
        path = nx.shortest_path(self.network,
                                source=self.current_node,
                                target=self.target_node)

        length = 0
        for n in range(len(path)-1):
            length += self.network.get_edge_data(path[n], path[n+1], 0)['length']

        return len(path), round(length/1000, 2)
      
  
    def reset(self, seed=None, return_info=False, options=None):
        """ Reset the environment and initiate a new episode

        options {
        'start_node': Index of the starting node for the agent
        'target_node': Index of the target node for the agent

        'start_time': Starting datetime object
        'deadline_time': Deadline datetime object

        'fuel': Initial fuel in tank
        }
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Use argument options
        if options:
            self.current_node = options['start_node']     # Index of current node
            self.target_node  = options['target_node']    # Index of client node
            
            self.current_time = options['start_time']    
            self.deadline     = options['deadline_time']
            
            # Fuel
            self.fuel         = options['fuel']

            path_steps, path_length = self._get_path_length()

        else:
            # Choose the agent's location uniformly at random
            self.current_node = self.np_random.integers(0, self.number_of_nodes)

            # We will sample the target's location randomly until it does not coincide
            # with the agent's location and exist a path between both nodes
            self.target_node = self.current_node
            while self.target_node == self.current_node or \
                    not nx.has_path(self.network, self.current_node, self.target_node):
                self.target_node = self.np_random.integers(0, self.number_of_nodes)

            # Depends on distance
            path_steps, path_length = self._get_path_length()

            # Exaggerated deadline: 1 minute per km
            self.current_time = datetime.strptime("06/11/21 09:30", "%d/%m/%y %H:%M")
            self.deadline     = self.current_time + timedelta(minutes=path_length)

            self.fuel = self.max_fuel


        # For plotting
        self.start_node = self.current_node

        # Auxiliary variables to calculate reward
        self.path = []

        self.seq_t  = 0   # Start HoS counters
        self.day_t  = 0
        self.week_t = 0

        # self.reward = 0
        # self.start_time = self.current_time

        self.total_hos_infractions  = 0
        self.total_tw_infractions   = 0
        self.total_fuel_infractions = 0

        self.total_fuel     = 0
        self.total_distance = 0
        self.total_duration = 0

        self.step_fuel      = 0
        self.step_distance  = 0
        self.step_duration  = 0

        # Compute neighbors and mask
        self.neighbors = list(self.network.neighbors(self.current_node))
        self.action_mask()

        # Get observation
        observation = self._get_obs()
        info = self._get_info()

        # Print information about the episode
        print(f"Departure point at node {self.current_node}")
        print(f"Target point at node {self.target_node}")
        print(f"Minimum length path of size {path_steps} with {path_length}km\n")

        return (observation, info) if return_info else observation

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------

    def step(self, action):
        # Update state
        self._apply_action(action)

        # Get mask of available actions
        self.action_mask()

        observation = self._get_obs()
        reward = self._get_reward()

        # An episode is done if the agent has reached the target
        done = True if self.current_node == self.target_node else False
        if done:
            print(f"Path found {self.path}\nDistance: {len(self.path)}")

        info = self._get_info()

        return observation, reward, done, info

    # ----------------------------------------------------------------------------

    # TODO: MOVE CURRENT_TIME PROPERLY
    def _apply_action(self, action):
        """ Update state according to action """
        # Check for valid action
        if action > self.REFUEL or action < 0:
          raise ValueError("Received invalid action={} which is not part of the action space".format(action))
    
        # If refuel, set fuel to max_fuel
        elif action == self.REFUEL:
            if (self.nodes[self.current_node]['type'] == self.TYPE_STATION or \
                self.nodes[self.current_node]['type'] == self.TYPE_PARKING_STATION) and \
                self._check_tw():
        
                self.fuel = self.max_fuel
                self.current_time += timedelta(minutes=self.REFUEL_TIME)

                # Update counters for reward
                self.step_distance = 0
                self.step_fuel = 0
                self.step_duration = self.REFUEL_TIME / 60   # h
      
            else: # TODO: Verify if not updating reward here could give problems (or give large negative)
                print("Refuel is not possible in this state")

        # If rest, set HoS counter to zero
        # TODO: Only rest if duration of rest does not surpass close time? Or clip duration?
        elif action >= self.REST_SEQ:
            if (self.nodes[self.current_node]['type'] == self.TYPE_PARKING or \
                self.nodes[self.current_node]['type'] == self.TYPE_PARKING_STATION) and \
                self._check_tw():

                # No matter the type of rest seq_t is always reset
                self.seq_t = 0
            
                if action == self.REST_SEQ:
                    dur = self.SEQ_REST

                elif action == self.REST_DAY:
                    self.day_t = 0

                    dur = self.DAILY_REST * 60

                else: # action == REST_WEEK:
                    self.day_t = 0
                    self.week_t = 0

                    dur = self.WEEKLY_REST * 60


                self.current_time += timedelta(minutes=dur)

                # Update counters for reward
                self.step_distance = 0
                self.step_fuel = 0
                self.step_duration = dur / 60    # h

            else:
                print("Rest is not possible in this state")

        # Check if exist edge
        elif not self.network.has_edge(self.current_node, action):
            print(f"Nodes {self.current_node} and {action} are not directly connected")
        
        # Otherwise move to node
        else:
            # Get selected edge
            # TODO: Passing key=0. Being a Multigraph, is possible that the returned
            # edge is not the correct one
            edge = self.network.get_edge_data(self.current_node, action, 0)

            length = 1 if not 'length' in edge else edge['length']  # attr always exist
            speed = 1 if not 'speed' in edge else edge['speed']
            # length = edge['length']
            # speed = edge['speed']
            duration = 1 / ((speed/60) / length) # in minutes

            # ------------------------------------------------------------------
            # Update counters for reward
            # fuel clipped to remaining fuel in tank
            self.step_distance = length / 1000                               # Km
            self.step_fuel = np.clip(self.FUEL_RATE * length, 0, self.fuel)  # L
            self.step_duration = duration / 60                               # h
            
            
            self.seq_t  += duration # HoS
            self.day_t  += duration
            self.week_t += duration

            self.fuel -= self.step_fuel  # Fuel consumption

            # ------------------------------------------------------------------
            # Update total infractions
            if self.fuel < 0:
              self.total_fuel_infractions += self.REGRET_FUEL
                    
            # HoS infractions
            if self.seq_t > self.MAX_SEQ_T:
              self.total_hos_infractions += self.REGRET_SEQT
            if self.day_t > self.MAX_DAY_T:
              self.total_hos_infractions += self.REGRET_DAYT
            if self.week_t > self.MAX_WEEK_T:
              self.total_hos_infractions += self.REGRET_WEET

            # Arriving in valid time window (we can assume being in target as 
            # otherwise value will be 0)
            self.total_tw_infractions += 0 if self._check_tw() else (self.REGRET_TW + self._find_tw_difference())

            # ------------------------------------------------------------------
            self.path.append(action)  # Store new node in path

            # Update current time
            self.current_time += timedelta(hours=duration//60,
                                            minutes=duration%60)

            # Compute neighbors
            self.neighbors = list(self.network.neighbors(action))

            self.current_node = action



        # Update total counters
        self.total_fuel += self.step_fuel
        self.total_duration += self.step_duration
        self.total_distance += self.step_distance

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def _get_reward(self):
        """
        Natural (informative) reward (negative, maximization problem). Calculated
        as:

        distance (km) + duration (h) + fuel (L) + infractions_penalty
        
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
        """
        # TODO: Round to Km, and hours?
        time_window_infraction = 0 if self._check_tw() else (self.REGRET_TW + self._find_tw_difference())

        fuel_infraction = self.REGRET_FUEL if self.fuel < 0 else 0

        hos_infraction = 0
        hos_infraction += self.REGRET_SEQT if self.seq_t > self.MAX_SEQ_T else 0
        hos_infraction += self.REGRET_DAYT if self.day_t > self.MAX_DAY_T else 0
        hos_infraction += self.REGRET_WEET if self.week_t > self.MAX_WEEK_T else 0

        reward = - (self.step_distance + self.step_duration + self.step_fuel + \
                    time_window_infraction + hos_infraction + fuel_infraction)

        return reward 

    def shaped_reward(self):
        """ 
        Shaped reward (RL agent will take a lot to update, NOT RECOMMENDED):
            - (length + duration + tw + fuel + HoS)    in client node
                - total_distance = 1 for each kilometer
                - total_duration = 1 for each minute
                - total_fuel     = 1 for each liter
                - total_tw_infractions   = Sum of (0 if arriving in time_window) (else 5 + 1 for each minute of difference) at each node
                - total_hos_infractions  = Sum of hos infractions at each node
                - total_fuel_infractions = Sum of fuel infractions at each node
            - zero otherwise
        """
        reward = 0

        if self.current_node == self.target_node: # They are index, no need fo np.array_equal()
            reward = (self.total_distance + self.total_duration + self.total_fuel +
                     self.total_hos_infractions +  self.total_tw_infractions + self.total_fuel_infractions)

        return reward

    # ----------------------------------------------------------------------------

    def action_mask(self):
        """ 
        Returns mask of available actions for the current state [1,0,1,...,1,1,1,0]
        That is: 
        - 1 for nodes with edge to 'current'
        - 1 for REST_SEQ, REST_DAY, REST_WEEK iif current=parking AND time in time_windows (could be soft constraint)
        - 1 for REFUEL iif current=station AND time in time_windows (could be soft constraint)
        - 0 otherwise
        """
        # Computing neighbors of current node
        self.mask = np.isin(self.nodes, self.neighbors,
                            assume_unique=True).astype(np.int64)

        # Check if node is parking and is open
        mask_parking = [0,0,0]
        if (self.nodes[self.current_node]['type'] == self.TYPE_PARKING or \
            self.nodes[self.current_node]['type'] == self.TYPE_PARKING_STATION) and \
            self._check_tw():
            mask_parking = [1,1,1]

        # Check if node is station and is open
        mask_refuel = 0
        if (self.nodes[self.current_node]['type'] == self.TYPE_STATION or \
            self.nodes[self.current_node]['type'] == self.TYPE_PARKING_STATION) and \
            self._check_tw():
            mask_refuel = 1
      
        self.mask = np.append(self.mask, mask_parking)
        self.mask = np.append(self.mask, mask_refuel)

        return self.mask

    # ----------------------------------------------------------------------------
    
    def action_space_sample(self):
        """ Samples an action available from the action space """
        if any(self.mask != 0):
            # [0] because np.where returns a tuple
            return np.int64(np.random.choice(np.where(self.mask == 1)[0]))
        else:
            print("No actions available!!")
            return self.action_space.sample()

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------

    def _check_tw(self):
        """ Check if current_time fits inside current_node time window """
        # Always True in normal node
        if self.nodes[self.current_node]['type'] == self.TYPE_NORMAL:
            return True
        else:
            h_open = self.nodes[self.current_node]['open_time'] // 60
            m_open = self.nodes[self.current_node]['open_time'] % 60

            h_clos = self.nodes[self.current_node]['close_time'] // 60
            m_clos = self.nodes[self.current_node]['close_time'] % 60

            return self.current_time.time() >= time(hour=h_open, minute=m_open) and \
                    self.current_time.time() < time(hour=h_clos, minute=m_clos)

    
    def _find_tw_difference(self):
        """ Number of hours between current time and current node tw """
        h_open = self.nodes[self.current_node]['open_time'] // 60
        m_open = self.nodes[self.current_node]['open_time'] % 60
        t_open = time(hour=h_open, minute=m_open)

        h_clos = self.nodes[self.current_node]['close_time'] // 60
        m_clos = self.nodes[self.current_node]['close_time'] % 60
        t_clos = time(hour=h_clos, minute=m_clos)
    
        a = b = 0
        if self.current_time.time() > t_clos:
            a = self.current_time.time() 
            b = t_clos

        elif self.current_time.time() < t_open:
            a = t_open
            b = self.current_time.time()

        # Calculate total seconds and rounding to minutes
        return int(timedelta(hours=(a.hour - b.hour), 
                            minutes=(a.minute - b.minute)).total_seconds() / 60)

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------

    def render(self, mode='human'):
        """ Plot map with current path, actual and target node at this step """    
        color_map = {0: 'w', 1: 'c', 2: 'y', 3: 'g'}

        fig, ax = ox.plot_graph_route(self.network, self.path,
                                      node_color=self.colors,
                                      figsize=(20,17),
                                      show=False, close=False) # OSMnx closes and
                                                               # shows by default
    
        # Plot start, current and target node
        self.gdf_nodes.iloc[
            [self.start_node, self.current_node, self.target_node]
        ].plot(ax=ax, color='lime', markersize=30)

        # Return
        if mode == 'rgb_array':
            return np.array(fig.canvas.buffer_rgba())
        elif mode == 'human':
            print(f"Total duration: {self.total_duration}")
            print(f"Total distance: {self.total_distance}")
            print(f"Total fuel: {self.total_fuel}")
            print(f"Path\n{self.path}")
            
            # TODO: print total_fuel consumed, distance, number of rest...

            plt.show()
            plt.close(fig)
        else:
            super().render(mode=mode) # just raise an exception