from WorldEvents import subscribe
import Agents
import random
import numpy as np
import math
import heapq

"""
WorldHandlers Module - Event response handlers for the Urban Catastrophe Simulation.

This module contains handler functions that respond to system events and modify
simulation state accordingly. It handles:
- Disaster initiation and casualty distribution
- Agent injury state transitions
- Emergency response dispatch coordination
- Dynamic system updates based on event triggers
- Cross-module state modifications

Handler functions receive event data as dictionaries and apply appropriate
changes to the World and Agent states.
"""
# To be populated with a World class reference upon initialization
world = any
gravely_injured_civilians: int = 0
healed_civilians: int = 0
dead_civilians: int = 0
safe_civilians: int = 0


def injure_near_disaster(data: dict) -> None:

    """
    Applies injury and death states to civilians within the disaster impact zone.
    
    Examines a 7x7 grid centered on the disaster location. Civilians in the 
    immediately adjacent cells (death radius) are killed instantly. Civilians 
    in the outer ring have a 50% chance of normal injury and 50% chance of 
    grave injury, with sick civilians automatically progressing to grave injury
    regardless of roll.
    
    Args:
        data: Dictionary containing:
            - world: World instance with the simulation state
            - disaster_location: Tuple (y, x) coordinates of disaster epicenter
    
    Side Effects:
        - Modifies health_state of affected civilian agents
        - Deaths create permanent obstacles in the grid
        - Caches world reference for future use
    """

    # extract data from dictionary
    global world
    world = data["world"]
    location: tuple = data["disaster_location"]

    # slice the map to get the area around the disaster site (disaster should end up at (3, 3))
    area_around_disaster: np.ndarray = world.map[location[0] - 3: location[0] + 4, location[1] - 3: location[1] + 4]

    # all the cells immediately next to the disaster site, any agent in one of these cells will die (RIP)
    death_radius: list[tuple] = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (3, 4), (4, 4), (4, 3), (3, 3)]

    # loop through all the cells surrounding the disaster, killing or injuring agents
    # nested loop to keep track of coordinates
    for i in range(len(area_around_disaster)):
        for j in range(len(area_around_disaster[i])):
            cell = area_around_disaster[i][j]

            # if a cell is a building, skip, cuz we can't kill or injure buildings
            if not cell.is_road:
                continue

            if cell.occupant is not None:
                # defensive redundency check, there should be nothing BUT civilians on the map at this point
                if isinstance(cell.occupant, Agents.Civilian):
                    # kills any civilians inside death radius
                    if (i, j) in death_radius:
                        cell.occupant.set_injury(Agents.Civilian.HealthState.DECEASED)

                    else:
                        chance: float = random.random()

                        # injures half of civilians outside blast radius
                        if chance <= 0.5:
                            cell.occupant.set_injury(Agents.Civilian.HealthState.INJURED)
                        
                        # gravely injures half of civilians outside blast radius
                        else:
                            cell.occupant.set_injury(Agents.Civilian.HealthState.GRAVELY_INJURED)

def dispatch_paramedic(data: dict) -> None:
    """
    Dispatches a paramedic to respond to a gravely injured civilian.
    
    Implements a three-tier dispatch strategy:
    1. If paramedics are available to spawn, creates a new paramedic at the 
       hospital closest to the victim and assigns them immediately.
    2. If all paramedics are spawned, queries existing paramedics from closest 
       to furthest, allowing each to accept/reject based on their queue capacity.
    3. If all paramedics reject (at capacity), forces assignment to a random 
       paramedic to ensure no civilian is abandoned.
    
    Args:
        data: Dictionary containing:
            - agent: The gravely injured Civilian requiring medical attention
    
    Side Effects:
        - May spawn new Paramedic and add to world.paramedics and world.agents
        - Updates paramedic's heal queue with the injured civilian
        - Sets perception for newly spawned paramedics
        
    Note:
        Requires cached world reference from disaster initialization.
        Uses Chebyshev distance for proximity calculations.
    """
    print("Trying paramedic dispatch")

    # gets the civilian information from the provided data
    agent = data["agent"]   

    #checks if we can still spawn more paramedics
    if len(world.paramedics) < world.num_paramedics:
        spawn_paramedic(agent)
    else:
        select_paramedic(agent)


def spawn_paramedic(agent):
    """
    Creates a new paramedic at the nearest hospital to respond to injury.
    
    Finds closest hospital spawn point to the injured civilian and attempts
    to spawn a paramedic there. Falls back to selection if spawn fails or
    no valid spawn locations exist.
    
    Args:
        agent: Civilian agent requiring medical attention
        
    Side Effects:
        - Creates new Paramedic instance
        - Adds paramedic to world.paramedics and world.agents
        - Sets initial perception for new paramedic
        - Falls back to select_paramedic if spawn fails
    """

    print("Trying paramedic spawn")
    agent_location: tuple = agent.location  
    # finds the closest spawn location to the civiliian
    sorted_spawn_locations: list[tuple[int, int]] = sorted(world.paramedic_spawn_locations, 
                                        key= lambda x: max(abs(x[0] - agent_location[0]), abs(x[1] - agent_location[1]))) 

    # spawns paramedic at nearest spawn location
    from Agents import Paramedic 

    def spawn_paramedic_inner():
        if len(sorted_spawn_locations) < 1:
            print("New paramedic spawn failedS")
            select_paramedic(agent)
        
        try:
            spawn_location = sorted_spawn_locations.pop(0)
            spawnable_cells = world.map[spawn_location[0] - 1: spawn_location[0] + 2, 
                                        spawn_location[1] - 1: spawn_location[1] + 2]
            new_paramedic = Paramedic(spawnable_cells, spawn_location, world.road_graph, agent)
        except:
            print("Paramedic failed to spawn, trying again")
            spawn_paramedic_inner()
        else:
            world.paramedics.append(new_paramedic)
            world.agents.append(new_paramedic)
            world.set_perception(new_paramedic)
            print("Paramedic successfully spawned")
    
    spawn_paramedic_inner()


def select_paramedic(agent):
    """
    Assigns an injured civilian to an existing paramedic's heal queue.
    
    Queries paramedics from closest to furthest, allowing each to accept
    based on queue capacity. If all reject, forces assignment to random
    paramedic to ensure coverage.
    
    Args:
        agent: Civilian agent requiring medical attention
        
    Side Effects:
        - Adds civilian to a paramedic's heal queue
        - May force assignment if all paramedics at capacity
    """
    print("Trying paramedic selection")

    if world.num_paramedics == 0:
        return

    agent_location: tuple = agent.location
    # makes a sorted copy of the list of all paramedics, sorted from closest to fartheset
    temp_paramedics = sorted(world.paramedics, key= lambda x: max(abs(x.location[0] - agent_location[0]), abs(x.location[1] - agent_location[1])))
    # keeps track of whether this agent has been assigned to a paramedic
    assigned_paramedic: Agents.Paramedic = None  #type: ignore


    #this function recursively 
    def ask_paramedic() -> None:
        nonlocal assigned_paramedic
        # checks if there are any more paramedics
        if len(temp_paramedics) == 0:
            return
        
        paramedic = temp_paramedics.pop(0)

        # tries to assign this agent to a paramedic
        accepted_assignment: bool = paramedic.add_to_heal_queue(agent)
        
        # if assignment is successful, then tell the outside
        if accepted_assignment:
            assigned_paramedic = paramedic
        # otherwise, check the next in line
        else:
            ask_paramedic()
            
    
    ask_paramedic()   

    if assigned_paramedic == None:
        # if no paramedic is available, forcibly assign this injured agent to a paramedic
        random_paramedic: Agents.Paramedic = random.choice(world.paramedics)

        distance_multiplier: float = 0.5
        health_multiplier: float = 1
        agent_time_to_worsen: float = agent.time_to_worsen
        distance_to_agent: float = max(abs(random_paramedic.location[0] - agent.location[0]), abs(random_paramedic.location[1] - agent.location[1]))

        agent_priority_score: float = (distance_multiplier * distance_to_agent) + (health_multiplier * agent_time_to_worsen)
        random_paramedic.counter += 1

        heal_queue_entry: tuple[float, int, Civilian] = (agent_priority_score, random_paramedic.counter, agent)
        heapq.heappush(random_paramedic.heal_queue, heal_queue_entry)

def count_safe_civilians(data: dict) -> None:
    """
    Increments counter for civilians who successfully escaped.
    
    Args:
        data: Dict containing 'agent' key with escaped Civilian
        
    Side Effects:
            - Increments global safe_civilians counter
    """

    global safe_civilians
    safe_civilians += 1

def count_dead_civilians(data: dict) -> None:
    """
    Increments counter for deceased civilians.
    
    Args:
        data: Dict containing 'agent' key with deceased Civilian
        
    Side Effects:
        - Increments global dead_civilians counter
    """
    global dead_civilians
    dead_civilians += 1

def count_gravely_injured_civilians(data: dict) -> None:
    """
    Increments counter for civilians who became gravely injured.
    
    Args:
        data: Dict containing 'agent' key with gravely injured Civilian
        
    Side Effects:
        - Increments global gravely_injured_civilians counter
    """
     
    global gravely_injured_civilians
    gravely_injured_civilians += 1

def count_healed_civilians(data: dict) -> None:
    """
    Increments counter for civilians successfully healed by paramedics.
    
    Args:
        data: Dict containing 'agent' key with healed Civilian
        
    Side Effects:
        - Increments global healed_civilians counter
    """

    global healed_civilians
    healed_civilians += 1

def calculate_stats(data: dict) -> None:
    """
    Prints final simulation statistics and calculates key metrics.
    
    Outputs total counts and calculates heal rate (healed/gravely injured)
    and survival rate (safe/(safe + dead)) as performance indicators.
    
    Args:
        data: Dict containing 'world' key with final World state
        
    Output:
        - Total counts for each civilian state
        - Heal rate percentage
        - Overall survival rate percentage
    """

    global gravely_injured_civilians 
    global dead_civilians
    global safe_civilians 
    global healed_civilians

    print(f"Civilians Gravely Injured: {gravely_injured_civilians}")
    print(f"Healed Civilians: {healed_civilians}")
    print(f"Dead Civilians: {dead_civilians}")
    print(f"Safe Civilians {safe_civilians}")

    print(" ")
    print(f"Heal Rate {healed_civilians / gravely_injured_civilians}")
    print(f"Survival rate {safe_civilians / (safe_civilians + dead_civilians)}")


# subscribe functions to events
def set_subscribe():
    """
    Registers all world handler functions to their respective events.
    
    Sets up the event-handler mapping for disaster response, medical
    dispatch, and statistics tracking. Must be called before simulation
    starts.
    
    Subscriptions:
        - disaster_start -> injure_near_disaster
        - help_needed -> dispatch_paramedic  
        - civilian safe -> count_safe_civilians
        - civilian gravely injured -> count_gravely_injured_civilians
        - civilian dead -> count_dead_civilians
        - civilian healed -> count_healed_civilians
        - simulation end -> calculate_stats
    """

    subscribe("disaster_start", injure_near_disaster)
    subscribe("help_needed", dispatch_paramedic)
    subscribe("civilian safe", count_safe_civilians)
    subscribe("civilian gravely injured", count_gravely_injured_civilians)
    subscribe("civilian dead", count_dead_civilians)
    subscribe("civilian healed", count_healed_civilians)
    subscribe("simulation end", calculate_stats)