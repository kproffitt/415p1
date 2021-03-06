# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  DEBUG = False
  if DEBUG == True: print "bob"
  if DEBUG == True: print [s,s,w,s,w,w,s,w]
  if DEBUG == True: print "bob"
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm 
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  """
  DEBUG = False;
  if DEBUG == True: print "Start:", problem.getStartState()
  if DEBUG == True: print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  if DEBUG == True: print "Start's successors:", problem.getSuccessors(problem.getStartState())
  "*** YOUR CODE HERE ***"
  """Create the frontier"""
  frontier = util.Stack()
  """Keep track of how you got somewhere"""
  frontierHash = {}
  """Keep track of explored nodes with True/False hashes"""
  exploredHash = {}
  """The current node being evaluated"""
  current = problem.getStartState()

  if problem.isGoalState(current):
    """Start is the goal"""
    return []
  """Otherwise, do some exploring"""
  frontier.push(current)
  exploredHash[current] = True
  frontierHash[current] = []
  """Do the search on the frontier"""
  while frontier.isEmpty() == False:
    current = frontier.pop()
    if DEBUG == True: print "Popping ",current,"off of frontier."
    if problem.isGoalState(current):
      """Return the path to the goal"""
      if DEBUG == True: print "The goal is:  ",current
      if DEBUG == True: print "The path is:  ",frontierHash[current]
      return frontierHash[current]
    
    successors = problem.getSuccessors(current)
    for successor in successors:
      """If the node hasn't been explored, put it on the frontier"""
      if successor[0] not in exploredHash:
	"""Add node to the frontier"""
        frontier.push(successor[0])
        """Add current to explored hashtable"""
        exploredHash[successor[0]] = True
	"""Add path to the node to the frontierHash"""
	path = list(frontierHash[current])
	path.append(successor[1])
	frontierHash[successor[0]] = path
	if DEBUG == True: print "Pushing ",successor[0]," at ",path

  util.raiseNotDefined()

def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
  "*** YOUR CODE HERE ***"
  DEBUG = False;

  """Create the frontier"""
  frontier = util.Queue()
  """Keep track of how you got somewhere"""
  frontierHash = {}
  """Keep track of explored nodes with True/False hashes"""
  exploredHash = {}
  """The current node being evaluated"""
  current = problem.getStartState()

  if problem.isGoalState(current):
    """Start is the goal"""
    return []
  """Otherwise, do some exploring"""
  frontier.push(current)
  exploredHash[current] = True
  frontierHash[current] = []
  """Do the search on the frontier"""
  while frontier.isEmpty() == False:
    current = frontier.pop()
    if DEBUG == True: print "Popping ",current,"off of frontier."
    if problem.isGoalState(current):
      """Return the path to the goal"""
      if DEBUG == True: print "The goal is:  ",current
      if DEBUG == True: print "The path is:  ",frontierHash[current]
      return frontierHash[current]
    
    successors = problem.getSuccessors(current)
    for successor in successors:
      """If the node hasn't been explored, put it on the frontier"""
      if successor[0] not in exploredHash:
	"""Add node to the frontier"""
        frontier.push(successor[0])
        """Add current to explored hashtable"""
        exploredHash[successor[0]] = True
	"""Add path to the node to the frontierHash"""
	path = list(frontierHash[current])
	path.append(successor[1])
	frontierHash[successor[0]] = path
	if DEBUG == True: print "Pushing ",successor[0]," at ",path

  util.raiseNotDefined()
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  DEBUG = False;
  """Create the frontier"""
  frontier = util.PriorityQueue()
  """Keep track of how you got somewhere"""
  frontierHash = {}
  """Keep track of explored nodes with True/False hashes"""
  exploredHash = {}
  """The current node being evaluated"""
  current = problem.getStartState()

  if problem.isGoalState(current):
    """Start is the goal"""
    return []
  """Otherwise, do some exploring.  Add arbitrary start priority"""
  frontier.push(current,0)
  exploredHash[current] = True
  frontierHash[current] = []
  """Do the search on the frontier"""
  while frontier.isEmpty() == False:
    current = frontier.pop()
    if DEBUG == True: print "Popping ",current,"off of frontier."
    if problem.isGoalState(current):
      """Return the path to the goal"""
      if DEBUG == True: print "The goal is:  ",current
      if DEBUG == True: print "The path is:  ",frontierHash[current]
      return frontierHash[current]
    
    successors = problem.getSuccessors(current)
    for successor in successors:
      """If the node hasn't been explored, put it on the frontier"""
      if successor[0] not in exploredHash:
	"""Add node to the frontier.  Use cost as priority"""
        frontier.push(successor[0],successor[2])
        """Add current to explored hashtable"""
        exploredHash[successor[0]] = True
	"""Add path to the node to the frontierHash"""
	path = list(frontierHash[current])
	path.append(successor[1])
	frontierHash[successor[0]] = path
	if DEBUG == True: print "Pushing ",successor[0]," at ",path
  util.raiseNotDefined()

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  """
  This is the same as the uniformCostSearch, except we change the cost
  going into the PriorityQueue.  Based on the video, it is calculated
  f=g+h where g(path) = path_cost and h(path)=h(s)=estimated_dist_to_goal

  We might need another hashtable or something to track the path cost like
  what we use to track the path to a particular node, I'm not sure yet.
  """
  DEBUG = False;
  """Create the frontier"""
  frontier = util.PriorityQueue()
  """Keep track of how you got somewhere"""
  frontierHash = {}
  """Keep track of explored nodes with True/False hashes"""
  exploredHash = {}
  """The current node being evaluated"""
  current = problem.getStartState()

  if problem.isGoalState(current):
    """Start is the goal"""
    return []
  """Otherwise, do some exploring.  Add arbitrary start priority"""
  frontier.push(current,0)
  exploredHash[current] = True
  frontierHash[current] = []
  """Do the search on the frontier"""
  while frontier.isEmpty() == False:
    current = frontier.pop()
    if DEBUG == True: print "Popping ",current,"off of frontier."
    if problem.isGoalState(current):
      """Return the path to the goal"""
      if DEBUG == True: print "The goal is:  ",current
      if DEBUG == True: print "The path is:  ",frontierHash[current]
      return frontierHash[current]
    
    successors = problem.getSuccessors(current)
    for successor in successors:
      """If the node hasn't been explored, put it on the frontier"""
      if successor[0] not in exploredHash:
	"""Add node to the frontier.  Calculate priroity with cost function for A-Star"""
        g=successor[2]
	"""Explanation:  nullHeuristic is passed into aStarSearch as a default
  parameter, relabled	as heuristic.  Therefore, unless something else is
	passed in the parameter, calling heuristic is the same as calling
	nullHeuristic."""
        h=heuristic(successor[0],problem)
        f = g+h
        """Add node to the frontier"""
        frontier.push(successor[0],f)
        """Add current to explored hashtable"""
        exploredHash[successor[0]] = True
	"""Add path to the node to the frontierHash"""
	path = list(frontierHash[current])
	path.append(successor[1])
	frontierHash[successor[0]] = path
	if DEBUG == True: print "Pushing ",successor[0]," at ",path
  util.raiseNotDefined()
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
