# searchAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
This file contains all of the agents that can be selected to 
control Pacman.  To select an agent, use the '-p' option
when running pacman.py.  Arguments can be passed to your agent
using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a searchFunction=depthFirstSearch

Commands to invoke other search strategies can be found in the 
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
# import self to use dir()
import searchAgents

class GoWestAgent(Agent):
  "An agent that goes West until it can't."
  
  def getAction(self, state):
    "The agent receives a GameState (defined in pacman.py)."
    if Directions.WEST in state.getLegalPacmanActions():
      return Directions.WEST
    else:
      return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
  """
  This very general search agent finds a path using a supplied search algorithm for a
  supplied search problem, then returns actions to follow that path.
  
  As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)
  
  Options for fn include:
    depthFirstSearch or dfs
    breadthFirstSearch or bfs
    
  
  Note: You should NOT change any code in SearchAgent
  """
    
  def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
    # Warning: some advanced Python magic is employed below to find the right functions and problems
    
    # Get the search function from the name and heuristic
    if fn not in dir(search): 
      raise AttributeError, fn + ' is not a search function in search.py.'
    func = getattr(search, fn)
    if 'heuristic' not in func.func_code.co_varnames:
      print('[SearchAgent] using function ' + fn) 
      self.searchFunction = func
    else:
      if heuristic in dir(searchAgents):
        heur = getattr(searchAgents, heuristic)
      elif heuristic in dir(search):
        heur = getattr(search, heuristic)
      else:
        raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
      print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic)) 
      # Note: this bit of Python trickery combines the search algorithm and the heuristic
      self.searchFunction = lambda x: func(x, heuristic=heur)
      
    # Get the search problem type from the name
    if prob not in dir(searchAgents) or not prob.endswith('Problem'): 
      raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
    self.searchType = getattr(searchAgents, prob)
    print('[SearchAgent] using problem type ' + prob) 
    
  def registerInitialState(self, state):
    """
    This is the first time that the agent sees the layout of the game board. Here, we
    choose a path to the goal.  In this phase, the agent should compute the path to the
    goal and store it in a local variable.  All of the work is done in this method!
    
    state: a GameState object (pacman.py)
    """
    if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
    starttime = time.time()
    problem = self.searchType(state) # Makes a new search problem
    self.actions  = self.searchFunction(problem) # Find a path
    totalCost = problem.getCostOfActions(self.actions)
    print('Path found with total cost of %d in %.2f seconds' % (totalCost, time.time() - starttime))
    if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)
    
  def getAction(self, state):
    """
    Returns the next action in the path chosen earlier (in registerInitialState).  Return
    Directions.STOP if there is no further action to take.
,corners
print "The walls are:  ",walls
    state: a GameState object (pacman.py)
    """
    if 'actionIndex' not in dir(self): self.actionIndex = 0
    i = self.actionIndex
    self.actionIndex += 1
    if i < len(self.actions):
      return self.actions[i]    
    else:
      return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
  """
  A search problem defines the state space, start state, goal test,
  successor function and cost function.  This search problem can be 
  used to find paths to a particular point on the pacman board.
  
  The state space consists of (x,y) positions in a pacman game.
  
  Note: this search problem is fully specified; you should NOT change it.
  """
  
  def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
    """
    Stores the start and goal.  
    
    gameState: A GameState object (pacman.py)
    costFn: A function from a search state (tuple) to a non-negative number
    goal: A position in the gameState
    """
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    if start != None: self.startState = start
    self.goal = goal
    self.costFn = costFn
    if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
      print 'Warning: this does not look like a regular search maze'

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
     isGoal = state == self.goal 
     
     # For display purposes only
     if isGoal:
       self._visitedlist.append(state)
       import __main__
       if '_display' in dir(__main__):
         if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
           __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable
       
     return isGoal   
   
  def getSuccessors(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """
    
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )
        
    # Bookkeeping for display purposes
    self._expanded += 1 
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)
      
    return successors

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999
    """
    if actions == None: return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x,y))
    return cost

class StayEastSearchAgent(SearchAgent):
  """
  An agent for position search with a cost function that penalizes being in
  positions on the West side of the board.  
  
  The cost function for stepping into a position (x,y) is 1/2^x.
  """
  def __init__(self):
      self.searchFunction = search.uniformCostSearch
      costFn = lambda pos: .5 ** pos[0] 
      self.searchType = lambda state: PositionSearchProblem(state, costFn)
      
class StayWestSearchAgent(SearchAgent):
  """
  An agent for position search with a cost function that penalizes being in
  positions on the East side of the board.  
  
  The cost function for stepping into a position (x,y) is 2^x.
  """
  def __init__(self):
      self.searchFunction = search.uniformCostSearch
      costFn = lambda pos: 2 ** pos[0] 
      self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
  "The Manhattan distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
  "The Euclidean distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
  """
  This search problem finds paths through all four corners of a layout.

  You must select a suitable state space and successor function
  """
  
  def __init__(self, startingGameState):
    """
    Stores the walls, pacman's starting position and corners.
    """
    self.walls = startingGameState.getWalls()
    self.startingPosition = startingGameState.getPacmanPosition()
    top, right = self.walls.height-2, self.walls.width-2 
    self.corners = ((1,1), (1,top), (right, 1), (right, top))
    for corner in self.corners:
      if not startingGameState.hasFood(*corner):
        print 'Warning: no food in corner ' + str(corner)
    self._expanded = 0 # Number of search nodes expanded
    
    "*** YOUR CODE HERE ***"
    self.debug=False
    
  def getStartState(self):
    "Returns the start state (in your state space, not the full Pacman state space)"
    "*** YOUR CODE HERE ***"
    if self.debug:
      print "Corners are: ",self.corners
    """visitedCorners keeps track of whether or not we've visited a corner.
    The position in visitedCorners matches the position of self.corners"""
    #visitedCorners = (False,False,False,False)
    """sp and sc to make less typing...this is just to make sure if we start
    on a corner we account for it as being visited"""
    sp = self.startingPosition
    sc = self.corners
    visitedCorners = ( sp == sc[0] , sp == sc[1] , sp == sc[2] , sp == sc[3] )
    if self.debug:
      print "visitedCorners=",visitedCorners
    """Each state will consist of its position in the graph AND 
    the tuple of visited corners"""
    state=self.startingPosition,visitedCorners
    if self.debug:
      print "Start state is: ",state
    return (state)
    util.raiseNotDefined()
    
  def isGoalState(self, state):
    if self.debug:
      print "Checking goal state...",state
    "Returns whether this search state is a goal state of the problem"
    "*** YOUR CODE HERE ***"
    visitedCorners = state[1]
    #return visitedCorners[0] and visitedCorners[1] and visitedCorners[2] and visitedCorners[3]
    return all(visitedCorners)
    util.raiseNotDefined()
       
  def getSuccessors(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """

    if self.debug:
      print "Starting to get successors for...",state
    
    """Pull out current position from state"""
    currentPosition=state[0]

    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      # Add a successor state to the successor list if the action is legal
      # Here's a code snippet for figuring out whether a new position hits a wall:
      x,y = currentPosition
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      hitsWall = self.walls[nextx][nexty]

      if not hitsWall:
        nextLocation=(nextx,nexty)
        
        """sc, nl, and vc to make less typing...this determines if we would go to a corner or
        keeps it marked if we already had visited a given corner"""
        sc = self.corners
        nl = nextLocation
        vc = state[1]
        visitedCorners = ( vc[0] or nl == sc[0] , vc[1] or nl == sc[1] , vc[2] or nl == sc[2] , vc[3] or nl == sc[3] )
        if self.debug:
          print "Successor ",nl," visitedCorners=",visitedCorners
        """Add the new successor to the list of successors by creating a tuple containing
        the position of the successor, the move to get to it, and the cost of that move.
        This is per directions at the top of this function."""
        newPosition=(nextLocation,visitedCorners)
        newSuccessor=(newPosition,action,1)
        successors.append(newSuccessor);
  
      "*** YOUR CODE HERE ***"
      
      """
      For the current state, check north, south, east, west
      If it hits a wall, don't return because that move is invalid
      If it doesn't hit a wall, make a tuple for the move
      Add the valid move tuple to the list of states to return
      Return the list of valid moves
      """
      
    self._expanded += 1
    if self.debug:
      print "Returning successors...",successors
    return successors

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999.  This is implemented for you.
    """
    if actions == None: return 999999
    x,y= self.startingPosition
    for action in actions:
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
    return len(actions)


def cornersHeuristic(state, problem):
  #State is a tuple.  Looks like a position and visitedCorners
  #Problem is the maze.
  """
  A heuristic for the CornersProblem that you defined.
  
    state:   The current search state 
             (a data structure you chose in your search problem)
    
    problem: The CornersProblem instance for this layout.  
    
  This function should always return a number that is a lower bound
  on the shortest path from the state to a goal of the problem; i.e.
  it should be admissible (as well as consistent).
  """
  corners = problem.corners # These are the corner coordinates
  walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

  DEBUG=False

  if DEBUG: print state

  nearest=-1 #Start negative so we know if it's been set up yet or not
  location=state[0] #The (x,y) where we are in the current state
  visitedCorners=state[1] #The list of visited corners
  """
  Loop through all the corners, and see what the distance to the nearest
  unvisited corner is from the current position (location)
  """
  for x in range(4):
    """Ignore corners we already visited, we only care about where we still need to go."""
    if not visitedCorners[x]:
      """Manhattan distance:  Distance between two points is strictly based on
      horizontal and vertical moves.  There are no diagonal movements allowed."""
      distance = abs(location[0]-corners[x][0])+abs(location[1]-corners[x][1])
      """Euclidean distance:  Distance between two points is measured as if with
      a ruler.  Diagonal movements are permitted, but this heuristic is not
      suitable for block-by-block movements where the moves are limited to up and down,
      left and right."""
      #distance = ((location[0] - corners[x][0]) ** 2 + (location[1] - corners[x][1]) ** 2) ** .5
      if DEBUG: print "Distance to ",corners[x]," is ",distance
      if distance < nearest or nearest == -1:
        nearest=distance/6 # a factor of 6 seems to give the best expanded node count 

  """Ensure non-negative response"""
  if nearest < 0:
    nearest=0
  if DEBUG: print "Nearest is ",nearest," moves"
  return nearest;

  

  "*** YOUR CODE HERE ***"
  #return 0 # Default to trivial solution

class AStarCornersAgent(SearchAgent):
  "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
  def __init__(self):
    self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
    self.searchType = CornersProblem

class FoodSearchProblem:
  """
  A search problem associated with finding the a path that collects all of the 
  food (dots) in a Pacman game.
  
  A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
    pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
    foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food 
  """
  def __init__(self, startingGameState):
    self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
    self.walls = startingGameState.getWalls()
    self.startingGameState = startingGameState
    self._expanded = 0
    self.heuristicInfo = {} # A dictionary for the heuristic to store information
      
  def getStartState(self):
    return self.start
  
  def isGoalState(self, state):
    return state[1].count() == 0

  def getSuccessors(self, state):
    "Returns successor states, the actions they require, and a cost of 1."
    successors = []
    self._expanded += 1
    for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state[0]
      dx, dy = Actions.directionToVector(direction)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextFood = state[1].copy()
        nextFood[nextx][nexty] = False
        successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
    return successors

  def getCostOfActions(self, actions):
    """Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999"""
    x,y= self.getStartState()[0]
    cost = 0
    for action in actions:
      # figure out the next state and see whether it's legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]:
        return 999999
      cost += 1
    return cost

class AStarFoodSearchAgent(SearchAgent):
  "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
  def __init__(self):
    self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
    self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
  """
  Your heuristic for the FoodSearchProblem goes here.
  
  This heuristic must be consistent to ensure correctness.  First, try to come up
  with an admissible heuristic; almost all admissible heuristics will be consistent
  as well.
  
  If using A* ever finds a solution that is worse uniform cost search finds,
  your heuristic is *not* consistent, and probably not admissible!  On the other hand,
  inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
  
  The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a 
  Grid (see game.py) of either True or False. You can call foodGrid.asList()
  to get a list of food coordinates instead.
  
  If you want access to info like walls, capsules, etc., you can query the problem.
  For example, problem.walls gives you a Grid of where the walls are.
  
  If you want to *store* information to be reused in other calls to the heuristic,
  there is a dictionary called problem.heuristicInfo that you can use. For example,
  if you only want to count the walls once and store that value, try:
    problem.heuristicInfo['wallCount'] = problem.walls.count()
  Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
  """
  position, foodGrid = state
  "*** YOUR CODE HERE ***"
  
  DEBUG=False

  if DEBUG: print state

  theFood=state[1].asList() #A list of tuple coordinates where food is
  nearest=-1 #Start negative so we know if it's been set up yet or not
  location=state[0] #The (x,y) where we are in the current state
  """
  Loop through all the corners, and see what the distance to the nearest
  unvisited corner is from the current position (location)
  """
  for x in range(len(theFood)):
    """Manhattan distance:  Distance between two points is strictly based on
    horizontal and vertical moves.  There are no diagonal movements allowed."""
    #distance = abs(location[0]-theFood[x][0])+abs(location[1]-theFood[x][1])
    """Euclidean distance:  Distance between two points is measured as if with
    a ruler.  Diagonal movements are permitted, but this heuristic is not
    suitable for block-by-block movements where the moves are limited to up and down,
    left and right."""
    distance = ((location[0] - theFood[x][0]) ** 2 + (location[1] - theFood[x][1]) ** 2) ** .5
    if DEBUG: print "Distance to ",theFood[x]," is ",distance
    if distance < nearest or nearest == -1:
      nearest=distance/7 # a factor of 7 seems to give the best expanded node count 

  """Ensure non-negative response"""
  if nearest < 0:
    nearest=0
  if DEBUG: print "Nearest is ",nearest," moves"
  return nearest

  #return 0 #Trivial solution
  
class ClosestDotSearchAgent(SearchAgent):
  "Search for all food using a sequence of searches"
  def registerInitialState(self, state):
    self.actions = []
    currentState = state
    while(currentState.getFood().count() > 0): 
      nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
      self.actions += nextPathSegment
      for action in nextPathSegment: 
        legal = currentState.getLegalActions()
        if action not in legal: 
          t = (str(action), str(currentState))
          raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
        currentState = currentState.generateSuccessor(0, action)
    self.actionIndex = 0
    print 'Path found with cost %d.' % len(self.actions)
    
  def findPathToClosestDot(self, gameState):
    "Returns a path (a list of actions) to the closest dot, starting from gameState"
    # Here are some useful elements of the startState
    startPosition = gameState.getPacmanPosition()
    food = gameState.getFood()
    walls = gameState.getWalls()
    problem = AnyFoodSearchProblem(gameState)

    "*** YOUR CODE HERE ***"
    print "gameState:\n"
    print type(gameState),"\n"
    print gameState,"\n"
    print "startPosition:\n"
    print type(startPosition),"\n"
    print startPosition,"\n"
    print "food:\n"
    print type(food),"\n"
    print food,"\n"
    print "walls:\n"
    print type(walls),"\n"
    print walls,"\n"
    print "problem:\n"
    print type(problem),"\n"
    print problem,"\n"
    
    """I think this is the right way to call a search, and it should return
    the path to the goal.  Now we need to try and figure out what the goal
    is and it should work.  I really want to use aStarSearch because I feel
    like that might be the most efficient thing to do but have to figure
    out how to make it work before I can worry about which type of search
    is the most efficient algorythm.  Also not sure if we have everything
    to call aStarSearch so I guess the uniformCostSearch will be a good
    place to start."""
    return search.uniformCostSearch(problem)

    util.raiseNotDefined()
  
class AnyFoodSearchProblem(PositionSearchProblem):
  """
    A search problem for finding a path to any food.
    
    This search problem is just like the PositionSearchProblem, but
    has a different goal test, which you need to fill in below.  The
    state space and successor function do not need to be changed.
    
    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.
    
    You can use this search problem to help you fill in 
    the findPathToClosestDot method.
  """

  def __init__(self, gameState):
    "Stores information from the gameState.  You don't need to change this."
    # Store the food for later reference
    self.food = gameState.getFood()

    # Store info for the PositionSearchProblem (no need to change this)
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0
    
  def isGoalState(self, state):
    """
    The state is Pacman's position. Fill this in with a goal test
    that will complete the problem definition.
    """
    x,y = state #What is the purpose of this line?
    """
                I assume that state is a tuple, so that
                stores the value of state[0] in x and
                state[1] in y but I can only guess that
                x and y might be coordinates because I
                don't know what the state contains.
                It doesn't look like they're ever used
                so I would say they have no purpose yet.
    """
    "*** YOUR CODE HERE ***"
    print "------------IN THE GOAL TEST--------------"
    """No idea if this works yet."""
    goalState = self.goal
    if state == goalState:
      return True
    else:
      return False

    """
    So, it seems good but self.goal apparently isn't defined.  I think
    we need to figure out how to say "if pacman x,y is sitting on a
    dot of food, then return true" but I'm having trouble figuring out
    what the "food" object is and what I can do to figure out if the
    spot at some given coordinate is food or not.
    """

    util.raiseNotDefined()

##################
# Mini-contest 1 #
##################

class ApproximateSearchAgent(Agent):
  "Implement your contest entry here.  Change anything but the class name."
  
  def registerInitialState(self, state):
    "This method is called before any moves are made."
    "*** YOUR CODE HERE ***"
    
  def getAction(self, state):
    """
    From game.py: 
    The Agent will receive a GameState and must return an action from 
    Directions.{North, South, East, West, Stop}
    """ 
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
def mazeDistance(point1, point2, gameState):
  """
  Returns the maze distance between any two points, using the search functions
  you have already built.  The gameState can be any game state -- Pacman's position
  in that state is ignored.
  
  Example usage: mazeDistance( (2,4), (5,6), gameState)
  
  This might be a useful helper function for your ApproximateSearchAgent.
  """
  x1, y1 = point1
  x2, y2 = point2
  walls = gameState.getWalls()
  assert not walls[x1][y1], 'point1 is a wall: ' + point1
  assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
  prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
  return len(search.bfs(prob))
