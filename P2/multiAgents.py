# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()
        ghostPosition = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = min(newScaredTimes)>0

        if not scared and (newPos in ghostPosition):
              return -1
        if newPos in currentGameState.getFood().asList():
          return 1
        
        closestFoodDist = sorted(newFood,key = lambda fDist: util.manhattanDistance(fDist, newPos))
        closestGhostDist = sorted(ghostPosition,key = lambda gDist: util.manhattanDistance(gDist, newPos))

        fdist = lambda fd: util.manhattanDistance(fd, newPos)
        fghost = lambda fg: util.manhattanDistance(fg, newPos)

      
        return 1.0/ fdist(closestFoodDist[0]) - 1.0 / fghost(closestGhostDist[0])

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        numGhosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, numGhosts)

    def maximize(self, gameState, depth, numGhosts):
        
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

        max_Value = float("-inf")
        best_action = Directions.STOP
        for action in gameState.getLegalActions(0):
          successor = gameState.generateSuccessor(0, action)
          temp_Value = self.minimize(successor, depth, 1, numGhosts)
          if temp_Value > max_Value:
            max_Value = temp_Value
            best_action = action

        
        if depth > 1:
          return max_Value
        return best_action

    def minimize(self, gameState, depth, agentIndex, numGhosts):
        
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        min_Value = float("inf")
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        if agentIndex == numGhosts:
          if depth < self.depth:
            for successor in successors:
              min_Value = min(min_Value, self.maximize(successor, depth + 1, numGhosts))
          else:
            for successor in successors:
              min_Value = min(min_Value, self.evaluationFunction(successor))
        else:
          for successor in successors:
            min_Value = min(min_Value, self.minimize(successor, depth, agentIndex + 1, numGhosts))
        return min_Value
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        GhostIndex = [i for i in range(1, gameState.getNumAgents())]
        inf = 1e100

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def min_value(state, d, ghost, A, B):

            if term(state, d):
                return self.evaluationFunction(state)

            v = inf
            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:  # next is maximizer with pacman
                    v = min(v, max_value(state.generateSuccessor(ghost, action), d + 1, A, B))
                else:  # next is minimizer with next-ghost
                    v = min(v, min_value(state.generateSuccessor(ghost, action), d, ghost + 1, A, B))

                if v < A:
                    return v
                B = min(B, v)

            return v

        def max_value(state, d, A, B):  # maximizer

            if term(state, d):
                return self.evaluationFunction(state)

            v = -inf
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), d, 1, A, B))

                if v > B:
                    return v
                A = max(A, v)

            return v

        def alphabeta(state):

            v = -inf
            act = None
            A = -inf
            B = inf

            for action in state.getLegalActions(0):  # maximizing
                tmp = min_value(gameState.generateSuccessor(0, action), 0, 1, A, B)

                if v < tmp:  # same as v = max(v, tmp)
                    v = tmp
                    act = action

                if v > B:  # pruning
                    return v
                A = max(A, tmp)

            return act

        return alphabeta(gameState)
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax_value (self, gameState, agentIndex, nodeDepth):

        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            nodeDepth += 1

        if nodeDepth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == self.index:
            return self.max_value(gameState, agentIndex, nodeDepth)
        else:
            return self.exp_value(gameState, agentIndex, nodeDepth)

        return 'None'

    def max_value(self, gameState, agentIndex, nodeDepth):

      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      value = float("-inf")
      actionValue = "Stop"

      for legalActions in gameState.getLegalActions(agentIndex):

        if legalActions == Directions.STOP:
          continue

        successor = gameState.generateSuccessor(agentIndex, legalActions)
        temp = self.expectimax_value(successor, agentIndex+1, nodeDepth)

        if temp > value:
          value = temp
          actionValue = legalActions

      if nodeDepth == 0:
        return actionValue
      else:
        return value

    def exp_value(self, gameState, agentIndex, nodeDepth):

      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      value = 0

      probValue = 1.0/len(gameState.getLegalActions(agentIndex))

      for legalActions in gameState.getLegalActions(agentIndex):
        if legalActions == Directions.STOP:
          continue

        successor = gameState.generateSuccessor(agentIndex, legalActions)
        temp = self.expectimax_value(successor, agentIndex+1, nodeDepth)

        value = value + (temp * probValue)
        actionValue = legalActions

      return value

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax_value(gameState, 0, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    curFoodList = currentGameState.getFood().asList()
    curFoodCount = currentGameState.getNumFood()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    curCapsules = currentGameState.getCapsules()
    curScore = currentGameState.getScore()

    foodLeft = 1.0/(curFoodCount + 1.0)
    ghostDist = float("inf")
    scaredGhosts = 0

    # print curScaredTimes

    for ghostState in curGhostStates:

      ghostPos = ghostState.getPosition()
      if curPos == ghostPos:
        return float("-inf")
      else:
        ghostDist = min(ghostDist,manhattanDistance(curPos,ghostPos))
      
      if ghostState.scaredTimer != 0:
        scaredGhosts += 1

    capDist = float("inf")
    for capsuleState in curCapsules:
      capDist = min(capDist,manhattanDistance(curPos,capsuleState))

    ghostDist = 1.0/(1.0 + (ghostDist/(len(curGhostStates))))
    # capDist = 1.0/(1.0 + capDist)
    capDist = 1.0/(1.0 + len(curCapsules))
    scaredGhosts = 1.0/(1.0 + scaredGhosts)


    return curScore + (foodLeft + ghostDist + capDist)

# Abbreviation
better = betterEvaluationFunction

