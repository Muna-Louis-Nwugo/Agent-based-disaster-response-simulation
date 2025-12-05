# Agent-based disaster response simulation
#### Agent-based simulation modeling emergent crowd behaviors and first-responder decision-making during large-scale urban disasters

For the past couple of months, I've been swimming in the shallow pools of simulation with my Wildfire Project, which gave me the confidence to try swimming in a small lake. I recently stumbled upon the Agent-based modeling lake on Google Maps, and it gave me the idea to rewind history a bit and ask myself, "What historical event would this kind of simulation have helped prepare for?".

This project was inspired by the 9/11 attacks on New York City. I had originally intended to test different Emergency Response systems and compare things like different communication protocols and centralized vs decentralized control, but I quickly realized that all of that might be just a _little_ outside my capabilities at the moment. So I scaled it back to something more focused: a system that models how first responders are dispatched and make decisions while also discovering interesting emergent behaviours that could occur when you have hundreds or thousands of civilians fleeing a single catastrophe. It's meant to be a general educational resource for anyone interested, though I would love to come back and build that original idea once my skills permit.

---
## Setup and Run Simulation

### Prerequisites
- Python
- NumPy
- Pygame

### Installation
Clone the repository:
```bash
git clone https://github.com/Muna-Louis-Nwugo/urban_catastrophe_simulation.git
cd urban_catastrophe_simulation
```

Install Dependencies
```bash
pip install numpy pygame
```

### Run Simulation
**Visual Mode (Pygame):**
```bash
python Render.py
```

**Console Mode (ASCII):**
```bash
python World.py
```

### Controls
- `SPACE` - Pause/unpause simulation
- Close window to exit

### Simulation Parameters (adjustable)
- **Grid Size**: 60x60 cells
- **Population**: 450 civilians, 5 paramedics
- **Disaster**: Triggers automatically at tick 300
- **Location**: (29, 25)
- **FPS**: 25

### Visual Indicators
- 🟩 Green - Healthy civilian
- 🟪 Purple - Sick civilian  
- 🟡 Yellow - Injured civilian
- 🔴 Red - Gravely injured civilian
- 🟤 Dark Red - Deceased civilian
- 🔵 Blue - Paramedic
- 🟠 Orange - Disaster zone
- 🔲 Gray - Buildings
- ⬛ Black - Roads

<img width="1079" height="1077" alt="image" src="https://github.com/user-attachments/assets/57118557-976f-485f-9af2-397083e46ab6" />

---

## System Architecture
I believe that systems should always be architected as if they were going to be extended, and this project is no different. I've developed a modular system that emphasizes extensibility, especially since I plan on returning and improving my project later. 
```
              Render
               ^  |
               |  V
    ---------- World <--- World Handlers
    |            |             ^
    V            V             |
  Agents -> World Events -------
                      
```

### Agents [The Beings]
- Contains agent classes, managing agent goals and behaviours (see Agent Behaviours)
- Agents are passed information about their surroundings from World Module
- Posts agent updates (e.g. injured or unfortunately deceased)

### World [The Environment]
- Contains the World class
- Map that stores Agent objects
- Passes perception data to Agents
- Handles tick-by-tick updates
- Sends information about world state to render
- Posts information about system updates (e.g. catastrophe initiated) to World Events

### World Events [The Messenger]
- Receives system updates from World and Agents, ferries update information to the appropriate World Handlers functions

### World Handlers [The Executioner]
- Updates specific parts of the system based on other system updates (e.g. Civilian injured -> check available paramedics -> dispatch paramedics)

### Render [The Display]
- Sends World Config data from the user to world
- Displays system state updates

Note: Both **Agents** and the **World** can emit events. These are routed through the **World Events** system and processed by the **World Handlers**, which apply the appropriate changes to the simulation state.

---
## Agent Behaviours
Each agent has different states and decision-making patterns to realistically simulate emergency scenarios.

### Civilian
**Patterns: Wander, Flee, Safe**
- Wander: Civilian wanders along roads from target to target on a map, taking into account how people go from one point to the next instead of aimlessly meandering
- Flee: Civilian establishes an exit (point on the edge of the map, preferably the closest, but not always) and flees towards that point at the maximum possible speed
- Safe: Civilian has successfully escaped

**States: Healthy, Sick, Injured, Gravely Injured, Deceased**
- Healthy: The civilian has no injuries (90% of civilians at the start)
- Sick: Civilian has an underlying condition (10% of civilians at the start)
  - Makes any sustained injures more severe (Sick civilians skip the Injured stage)
- Injured: Civilian that has been injured during catastrophe
- Gravely Injured: Second stage of injury, civilian's condition deteriorates rapidly
- Deceased: Civilian has died (RIP)

### Paramedic
**Patterns: Standby, Dispatched**
- Standby: There are no civilians for the paramedic to tend to. Paramedic returns to hospital.
- Dispatched: There is at least one civilian that the paramedic needs to tend to. Paramedic is working on saving these civilians.

###### Police and Firefighter Agents proved to require a lot of complexity to be even remotely accurate (perimeter establishment, crowd control, civilian transportation, etd.), so I've decided to skip them and try tackle them with the next iteration of this project.

---
## System Rules and Mechanics

### Disaster Impact (Tick 300)
When the catastrophe occurs at coordinates (29, 25):
- **Death Zone**: All civilians in immediately adjacent cells (8 cells surrounding disaster) die instantly
- **Injury Zone**: Civilians in the outer ring (7x7 grid minus death zone):
  - 50% chance of INJURED state
  - 50% chance of GRAVELY_INJURED state
  - Exception: SICK civilians always become GRAVELY_INJURED regardless of roll

### Health State Progression
Civilians deteriorate without medical intervention:
```
INJURED → (60 ticks) → GRAVELY_INJURED → (20 ticks) → DECEASED
```
- Only paramedics can halt deterioration

### Panic Spread Mechanics
Wandering civilians begin fleeing when they perceive:
1. The disaster itself within their 7x7 perception grid
2. More than 5 other civilians fleeing
3. More than 2 casualties (gravely injured or deceased)

Once fleeing, civilians have shared disaster knowledge and navigate to the nearest safe edge.

### Crowd Crush Dynamics
When fleeing, if 20+ agents surround a civilian:
- 5% chance per tick of sustaining injury
- Simulates trampling and crowd crush scenarios
- Dead civilians don't count toward crowd density

### Paramedic Response System
**Dispatch Priority**:
1. Spawn new paramedic at nearest hospital until capacity is reached
2. Assign to existing paramedic with a queue that isn't full
3. Force assignment to random paramedic if all queues full (5+ patients)

**Healing Priority Score**:
```
Priority = (0.5 × distance_to_victim) + (1.0 × time_remaining)
```
Lower scores = higher priority. Paramedics use min-heap for patient queue.

**Healing Rules**:
- Paramedics heal GRAVELY_INJURED → INJURED on contact
- Opportunistic healing: will heal any gravely injured civilian encountered en route

### Movement and Navigation
- All agents use A* pathfinding with Chebyshev distance heuristic
- 8-directional movement (including diagonals)
- Local avoidance when blocked: agents evaluate all 8 neighbors and pick closest to desired path
- Path recalculation triggered if agent drifts 4+ cells from planned route
- Agents cannot traverse buildings or occupy same cell

### Safe Zone Selection (Fleeing)
Civilians calculate escape routes away from disaster:
- Filter edge cells to only those that don't require moving toward disaster
- Select nearest valid edge by Chebyshev distance

---
## Technologies/Methods Used
- Python
- Numpy
- Agent-Based Modeling
- Pathfinding Algorithms
- Multi-Agent Coordination
- Object Oriented Programming
- Event Driven Architecture

---
## Project Status
### Completed
- Agent spawn
- Agent awareness
- Agent Pathfinding
- Agent pathfollowing and local avoidance
- Catastrophy initialization
- Civilian Patterns
- Civilian Health State Machine
- Paramedic Dispatch
-  Paramedic triage

### Planned
- Firefighter
- Police Officers
