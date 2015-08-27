# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
    return [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    from game import Directions
    from util import Stack
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    visited = []
    visitedSet = set(visited)
    myStack = util.Stack()
    levelStack = []
    pos = problem.getStartState()
    myStack.push((pos, None, 1))

    while not myStack.isEmpty():
        info = myStack.pop()
        visitedSet.add(info[0])

        if len(levelStack) < info[2]:
            levelStack.append(info[1])
        else:
            ind = info[2] - 1
            i = 0
            while not (i == ind):
                i = i + 1
            levelStack[i] = info[1]

        if problem.isGoalState(info[0]):
            if levelStack[0] == None:
                levelStack = levelStack[1:]
            return levelStack

        sucList = problem.getSuccessors(info[0]);
        for adj in sucList:
            if adj[0] not in visitedSet:
                if not (info[0] == pos):
                    newadj = (adj[0], adj[1], info[2] + 1)
                    myStack.push(newadj)
                else:
                    myStack.push(adj)            


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    #  "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    # from game import Directions
    # from util import Queue
    
    # s = Directions.SOUTH
    # w = Directions.WEST
    # e = Directions.EAST
    # n = Directions.NORTH
    # visited = []
    # visitedSet = set(visited)
    # myQueue = util.Queue()
    # pos = problem.getStartState()
    # visitedSet.add(pos)

    # parentDict = {}
    # parentDict[pos] = (pos, None)
    # myQueue.push((pos, None))

    # while not myQueue.isEmpty():
    #     info = myQueue.pop()
    #     visitedSet.add(info[0])
        
    #     if problem.isGoalState(info[0]):
    #         indTuple = info[0]
    #         toReturn = []
    #         while not (parentDict[indTuple][0] == indTuple):
    #             toReturn.insert(0, parentDict[indTuple][1])
    #             indTuple = parentDict[indTuple][0]
    #         toReturn = toReturn[1:]
    #         toReturn.append(info[1])
    #         print toReturn
    #         return toReturn

    #     sucList = problem.getSuccessors(info[0]);
    #     for adj in sucList:
    #         if adj[0] not in visitedSet:
    #             if not (adj[0] in parentDict.keys()):
    #                 myQueue.push(adj)
    #                 parentDict[adj[0]] = (info[0], info[1])
    #           

    from game import Directions
    from util import Queue
    
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    visited = []
    myQueue = util.Queue()
    pos = problem.getStartState()
    myQueue.push((pos, []));

    while not myQueue.isEmpty():
        pt, dirct = myQueue.pop()

        if problem.isGoalState(pt):
            return dirct

        if not pt in visited:
            sucList = problem.getSuccessors(pt);
            for adj, nextDir, dist in sucList:
                if not adj in visited: 
                    newDir = dirct + [nextDir]
                    myQueue.push((adj, newDir))

        visited.append(pt)


def uniformCostSearch(problem):
    # "Search the node of least total cost first. "
    # "*** YOUR CODE HERE ***"
    # # util.raiseNotDefined()
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    from game import Directions
    from util import PriorityQueue
    
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    visited = []
    visitedSet = set(visited)
    myPq = util.PriorityQueue()
    pos = problem.getStartState()
    myPq.push((pos, []), 0);

    while not myPq.isEmpty():
        pt, dirct = myPq.pop()
        
        if problem.isGoalState(pt):
            return dirct

        if pt not in visitedSet:
            sucList = problem.getSuccessors(pt);
            for adj, nextDir, dist in sucList:
                if adj not in visitedSet: 
                    newDir = dirct + [nextDir]
                    myPq.push((adj, newDir), problem.getCostOfActions(newDir))

        visitedSet.add(pt)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    from game import Directions
    from util import PriorityQueue
    
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    visited = []
    pos = problem.getStartState()
    myPq = util.PriorityQueue()
    myPq.push((pos, []), heuristic(pos, problem));

    while not myPq.isEmpty():

        pt, dirct = myPq.pop()
        
        if problem.isGoalState(pt):
            return dirct

        if pt not in visited:
            sucList = problem.getSuccessors(pt);
            for adj, nextDir, dist in sucList:
                if adj not in visited: 
                    newDir = dirct + [nextDir]
                    myPq.push((adj, newDir), heuristic(adj, problem) + problem.getCostOfActions(newDir))

        visited.append(pt)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
