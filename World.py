import Agents
import numpy as np
import random
import time
from WorldEvents import post
from WorldHandlers import set_subscribe

set_subscribe()


"""
World Module - Main simulation engine for the Urban Catastrophe Simulation.

This module contains the World class which manages the spatial environment, agent population, 
and simulation state. It handles:
- Grid-based city representation with roads, buildings, and intersections
- Agent spawning, movement, and collision detection
- Simulation tick updates and event dispatching
- Spatial queries for pathfinding and agent perception
- Communication between agents and environment

The World class serves as the central coordinator, maintaining the 100x100 cell grid
and orchestrating all agent behaviors during catastrophic events.
"""

class Cell():
    """
    The Cell class represents each cell in the 100x100 grid. It stores information about what type of cell it is
    (road or building), whether the cell is a disaster site, and which agent is currently occupying the cell.
    """
    def __init__(self, is_road: bool, disaster: bool = False):
        """
        Initializes a grid cell with terrain type and disaster status.
        
        Args:
            is_road: True if cell is traversable road, False if building
            disaster: True if cell is disaster epicenter, False otherwise
        """

        self.is_road: bool = is_road
        self.disaster: bool = disaster
        self.occupant: Agents.Agent = None  # type: ignore

class World():
    """
    The World class acts as the main simulation engine. It harbors the map agents traverse, the agents themselves,
    And the information that every agent needs to function properly.

    Properties: 
        num_civilians -> Number of civilians on the map
        num_paramedics -> Number of paramedics on the map
        cell_occupants -> Which cell is occupied and by which agent
        map -> the grid map that the agents traverse along (numpy array of cells)
        road_graph -> a graphical representation of the grid map, except only including roads. Enables pathfinding around buildings
        disaster_loc -> the grid-coordinate location of the catastrophe
        agents -> list of all agents
        wall -> a cell object that represents out-of-grid cells. Cached for self.set_perception
        paramedics -> a complete list of all the paramedics currently on the map
        paramedic_spawn_locations -> list of paramedic spawn locations

        #Note: the grid map is to be made up of a 2d numpy array of Cell objects, to help each cell store data more effectively
    """

    def __init__(self, num_civilians: int, num_paramedics: int, map: np.ndarray, paramedic_spawn_locations: list[tuple[int, int]] = [(20, 20), (20, 40), (40, 30)]):
        self.num_civilians: int = num_civilians
        self.num_paramedics: int = num_paramedics
        self.map: np.ndarray = map 
        self.road_graph: dict = self.init_road_graph()
        self.disaster_loc: tuple = None #type: ignore
        self.agents: list[Agents.Agent] = []
        self.wall = Cell(False)
        self.paramedics: list[Agents.Paramedic] = []
        self.paramedic_spawn_locations: list[tuple[int, int]] = paramedic_spawn_locations

        self.civilian_spawn(self.num_civilians)

        for y, x in self.paramedic_spawn_locations:
            if not self.map[y][x].is_road:
                raise ValueError(f"Hospital at {y, x} is situated on a building. Please place it on a road")

    
    
    # Return a hashmap of this world's traversible cells
    def init_road_graph(self) -> dict:
        """
        Constructs an adjacency graph of all traversable road cells for pathfinding.
        
        Examines each cell in the grid and, for road cells, identifies all valid 
        neighboring road cells within a 1-cell radius (8-directional movement). 
        The resulting graph maps each road cell's coordinates to a list of 
        accessible neighbor coordinates.
        
        Returns:
            dict: Adjacency graph where keys are road cell coordinates (y, x) and 
                values are lists of neighboring road cell coordinates that can 
                be reached in one move.
                
        Example:
            {(0, 1): [(0, 2), (1, 1), (1, 2)],
            (0, 2): [(0, 1), (0, 3), (1, 2)], ...}
        """
        # container for our graph, to be filled in
        graph: dict = {}
        
        #total possible neighbours around a cell
        possible_neighbours: list[tuple[int, int]] = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1), (1, -1), (0, -1)]

        #looping through cells, populating graph
        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if self.map[y][x].is_road:
                    #All valid
                    valid_neighbours = []

                    for dy, dx in possible_neighbours:
                        ny, nx = dy + y, dx + x

                        if ny < self.map.shape[0] and nx < self.map.shape[1] and ny >= 0 and nx >= 0:
                            if self.map[ny][nx].is_road:
                                valid_neighbours.append((ny, nx))
                    

                    graph[(y, x)] = valid_neighbours
        
        return graph

    
    # EFFECT: Spawns civilians in the grid
    def civilian_spawn(self, num_civilians: int) -> None:
        """
        Spawns the specified number of civilians randomly on available road cells.
        
        Iterates through random road positions until all agents are placed. Ensures
        no two agents occupy the same cell during initialization. Each agent receives
        its initial perception and movement options based on spawn location.
        
        Args:
            num_civilians: Number of agents to spawn
        
        Side Effects:
            - Adds civilians to self.agents list
            - Updates cell occupancy in self.map
            - Sets initial perception for each spawned civilian
            - Sets 10% of spawned civilians to the "SICK" health state
        """

        road_list: list[tuple] = list(self.road_graph.keys())

        while num_civilians > 0:
            rand_num = random.randrange(0, len(road_list))
            desired_cell = road_list[rand_num]

            if self.map[desired_cell[0], desired_cell[1]].occupant is not None: # type: ignore #checks if a cell is already occupied
                continue
            else:
                new_civilian: Agents.Civilian = Agents.Civilian(desired_cell, self.road_graph)
                self.set_perception(new_civilian)
                self.map[desired_cell[0], desired_cell[1]].occupant = new_civilian #type: ignore
                self.agents.append(new_civilian)

                if random.random() <= 0.1:
                    new_civilian.health_state = Agents.Civilian.HealthState.SICK

                num_civilians -= 1      


    # EFFECT: initializes agent perception
    def set_perception(self, agent) -> None:
        """
        Updates an agent's perception array with their surrounding environment.
        
        Extracts a 7x7 grid centered on the agent's current position, giving them
        visibility 3 cells in each direction. Handles edge cases where agents are
        near map boundaries by numpy's automatic bounds clipping.
        
        Args:
            agent: The agent whose perception needs updating
        
        Side Effects:
            - Updates agent.perception with current surrounding grid
        """
        y, x = agent.location
        
        # Track the actual slice boundaries
        y_start:int = max(0, y - 3)
        y_end: int = min(self.map.shape[0], y + 4)
        x_start: int = max(0, x - 3)
        x_end: int = min(self.map.shape[1], x + 4)
        
        perception = self.map[y_start:y_end, x_start:x_end]

        y_start_pad: int = 0
        y_end_pad: int = 0
        x_start_pad: int = 0
        x_end_pad: int = 0

        if y_start == 0:
            y_start_pad = -(y-3)
        if x_start == 0:
            x_start_pad = -(x-3)
        if y_end == self.map.shape[0]:
            y_end_pad = (y+4) - self.map.shape[0]
        if x_end == self.map.shape[1]:
            x_end_pad = (x+4) - self.map.shape[1]

        padded_perception = np.pad(perception, ((y_start_pad, y_end_pad), (x_start_pad, x_end_pad)), mode="constant", constant_values= self.wall) # type: ignore
        
        agent.perception = padded_perception


    #sets the location of a disaster
    def set_disaster_loc(self, loc: tuple):
        """
        Activates disaster at specified location and alerts all systems.
        
        Marks the disaster cell, updates global agent disaster knowledge,
        and posts event to trigger injury calculations and panic spread.
        
        Args:
            loc: Tuple (y, x) coordinates of disaster epicenter
            
        Side Effects:
            - Sets cell.disaster to True at location
            - Updates Agent.disaster_loc class variable
            - Posts "disaster_start" event with world and location
        """
         
        self.disaster_loc = loc
        self.map[loc[0]][loc[1]].disaster = True #type: ignore
        Agents.Agent.disaster_loc = loc
        post("disaster_start", {"world": self, "disaster_location": loc})


    #EFFECT: updates every agent on the grid
    def update(self):
        """
        Executes one simulation tick, updating all agent positions and states.
        
        For each agent: captures current position, calls agent's update method
        (which may change position), updates grid occupancy, refreshes perception
        based on new position, and updates available movement options. Does not
        handle collision detection in current implementation.
        
        Side Effects:
            - Refreshes agent perceptions
            - Updates all agent positions
            - Updates cell occupancy in self.map
        
        INTERESTING TEST:
            Traffic tends to cluster near the upper left corner of the map, does shuffling the order of agent operations on each tick change that?
            Answer, yes. processing order was biased towards the left, after I implemented the change, traffic starte concentrating towards the upper center of the map.
        """

        random.shuffle(self.agents)

        for agent in self.agents:
            self.set_perception(agent)

            old_loc = agent.location
            agent.update()
            new_loc = agent.location

            self.map[old_loc[0], old_loc[1]].occupant = None # type: ignore

            if agent.pattern is not Agents.Civilian.Pattern.SAFE: #type: ignore
                self.map[new_loc[0], new_loc[1]].occupant = agent # type: ignore


    def draw(self) -> None:
        """
        Renders the current simulation state to console using ASCII characters.
        
        Displays a grid representation where:
        - '█' represents buildings (non-traversable)
        - ' ' represents empty roads (space for better visibility)
        - 'h' represents healthy civilians
        - 's' represents sick civilians
        - 'i' represents injured civilians
        - 'G' represents gravely injured civilians
        - 'D' represents deceased civilians
        - 'P' represents paramedics
        - 'X' represents disaster location
        """
        print("\n" + "="*50)  # Separator line
        print("  ", end="")
        
        # Print column numbers (every 5th for readability)
        for x in range(len(self.map[0])):
            if x % 5 == 0:
                print(f"{x:2}", end="")
            else:
                print("  ", end="")
        print()
        
        # Print each row
        for y in range(len(self.map)):
            print(f"{y:2} ", end="")  # Row number with padding
            
            for x in range(len(self.map[y])):
                cell = self.map[y][x]
                
                # Check if this is disaster location
                if cell.disaster:
                    print("X ", end="")
                elif not cell.is_road:
                    print("█ ", end="")
                elif cell.occupant is None:
                    print("  ", end="")  # Empty space instead of dot
                elif isinstance(cell.occupant, Agents.Paramedic):
                    print("P ", end="")  # Paramedic
                elif isinstance(cell.occupant, Agents.Civilian):
                    health = cell.occupant.health_state
                    if health == Agents.Civilian.HealthState.HEALTHY:
                        print("h ", end="")
                    elif health == Agents.Civilian.HealthState.SICK:
                        print("s ", end="")
                    elif health == Agents.Civilian.HealthState.INJURED:
                        print("i ", end="")
                    elif health == Agents.Civilian.HealthState.GRAVELY_INJURED:
                        print("G ", end="")
                    elif health == Agents.Civilian.HealthState.DECEASED:
                        print("D ", end="")
                    else:
                        print("? ", end="")
                else:
                    print("? ", end="")
            print()

if __name__ == "__main__": 
    # Generate a 70x70 city grid with multi-lane roads
    size = 60
    test_grid = [[False for _ in range(size)] for _ in range(size)]
    
    # Create main avenues (3 lanes wide) every 10 blocks
    for i in range(0, size, 10):
        for j in range(size):
            # Vertical avenues
            if i < size - 2:
                test_grid[j][i] = True
                test_grid[j][i+1] = True
                test_grid[j][i+2] = True
            # Horizontal avenues
            if j < size - 2:
                test_grid[i][j] = True
                test_grid[i+1][j] = True
                test_grid[i+2][j] = True
    
    # Add smaller streets (2 lanes wide) between avenues
    for i in range(5, size, 10):
        for j in range(size):
            # Vertical streets
            if i < size - 1:
                test_grid[j][i] = True
                test_grid[j][i+1] = True
            # Horizontal streets
            if j < size - 1:
                test_grid[i][j] = True
                test_grid[i+1][j] = True

    # Convert to numpy array of Cell objects
    map_array = np.empty((size, size), dtype=object)
    for y in range(size):
        for x in range(size):
            map_array[y, x] = Cell(test_grid[y][x])

    # Create world
    world = World(num_civilians=450, num_paramedics=5, map=map_array) #type: ignore

    for i in range(300):
        start = time.time()
        world.update()
        print(f"Update took: {time.time() - start:.3f} seconds")
        #world.draw()
        #time.sleep(0.05)
    
    print("CATASTROPHE COMMENCED")
    world.set_disaster_loc((29, 25))
    
    for i in range(600):
        start = time.time()
        world.update()
        print(f"Update took: {time.time() - start:.3f} seconds")
        #world.draw()
    
    post("simulation end", {"world": world})

    
    """ import time
    start = time.time()
    world.update()
    print(f"Update took: {time.time() - start:.3f} seconds") """