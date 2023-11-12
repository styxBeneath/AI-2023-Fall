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
import math
import random
import util

from game import Agent


def euclideanDistance(xy1, xy2):
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


class ReflexAgent(Agent):
    def food_multiplier(self, food_grid, position):
        food_count = 0
        closest_distance = float('inf')

        for i in range(food_grid.height):
            for j in range(food_grid.width):
                if food_grid[j][i]:
                    food_count += 1
                    curr_point = (j, i)
                    curr_distance = util.manhattanDistance(position, curr_point)
                    closest_distance = min(closest_distance, curr_distance)

        return 100 / max(closest_distance, 1) if food_count > 0 else 0

    def ghost_multiplier(self, new_position, new_ghost_states, new_scared_times):
        danger = 35
        ghost = new_ghost_states[0]
        distance_to_ghost = euclideanDistance(new_position, ghost.getPosition())

        if new_scared_times[0] == 0 and distance_to_ghost <= 4:
            danger += 160 * (8 - distance_to_ghost)

        return danger

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        return (self.food_multiplier(currentGameState.getFood(), newPos)
                - self.ghost_multiplier(newPos, newGhostStates, newScaredTimes))


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def get_min_value(self, game_state, depth, index):
        val = math.inf

        new_index = (index + 1) % game_state.getNumAgents()
        new_depth = depth
        if new_index == 0:
            new_depth -= 1

        legal_actions = game_state.getLegalActions(index)
        for action in legal_actions:
            successor = game_state.generateSuccessor(index, action)
            successor_value = self.get_value(successor, new_depth, new_index)
            val = min(val, successor_value)
        return val

    def get_max_value(self, game_state, depth):
        val = -math.inf
        legal_actions = game_state.getLegalActions(0)
        best_action = legal_actions[0]
        for action in legal_actions:
            successor = game_state.generateSuccessor(0, action)
            successor_value = self.get_value(successor, depth, 1)
            if successor_value > val:
                best_action = action
                val = successor_value
        return val, best_action

    def get_value(self, game_state, depth, index):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)
        if index == 0:
            return self.get_max_value(game_state, depth)[0]
        return self.get_min_value(game_state, depth, index)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self. Depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.get_max_value(gameState, self.depth)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    def min_value(self, game_state, depth, index, a, b):
        val = math.inf
        new_index = (index + 1) % game_state.getNumAgents()
        new_depth = depth
        if new_index == 0:
            new_depth -= 1

        valid_actions = game_state.getLegalActions(index)
        for action in valid_actions:
            successor = game_state.generateSuccessor(index, action)
            successor_value = self.value(successor, new_depth, new_index, a, b)
            val = min(val, successor_value)
            if val < a:
                return val
            b = min(b, val)
        return val

    def max_value(self, game_state, depth, a, b):
        val = -math.inf
        valid_actions = game_state.getLegalActions(0)
        best_action = valid_actions[0]
        for action in valid_actions:
            successor = game_state.generateSuccessor(0, action)
            successor_value = self.value(successor, depth, 1, a, b)
            if successor_value > val:
                best_action = action
                val = successor_value
            if val > b:
                return val, best_action
            a = max(a, val)
        return val, best_action

    def value(self, game_state, depth, index, a, b):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)
        if index == 0:
            return self.max_value(game_state, depth, a, b)[0]
        return self.min_value(game_state, depth, index, a, b)

    def getAction(self, gameState):
        """
        Returns the minimax action using self. Depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a = -math.inf
        b = math.inf
        return self.max_value(gameState, self.depth, a, b)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    def max_value(self, game_state, depth):
        val = -math.inf
        valid_actions = game_state.getLegalActions(0)
        best_action = valid_actions[0]
        for action in valid_actions:
            successor = game_state.generateSuccessor(0, action)
            successor_value = self.value(successor, depth, 1)
            if successor_value > val:
                best_action = action
                val = successor_value
        return val, best_action

    def expected_value(self, game_state, depth, index):
        val = 0

        new_index = (index + 1) % game_state.getNumAgents()
        new_depth = depth
        if new_index == 0:
            new_depth -= 1

        valid_actions = game_state.getLegalActions(index)
        for action in valid_actions:
            successor = game_state.generateSuccessor(index, action)
            successor_value = self.value(successor, new_depth, new_index)
            prob = 1 / len(valid_actions)
            val += prob * successor_value
        return val

    def value(self, game_state, depth, index):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)
        if index == 0:
            return self.max_value(game_state, depth)[0]
        return self.expected_value(game_state, depth, index)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self. Depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, self.depth)[1]


def food_multiplier(game_state):
    return 70000 / max(1, game_state.getNumFood())


def ghost_multiplier(game_state):
    ghost_states = game_state.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    position = game_state.getPacmanPosition()

    danger = 0
    euclidean_dist_to_ghost = euclideanDistance(position, ghost_states[0].getPosition())
    if scared_times[0] > 0:
        if scared_times[0] > euclidean_dist_to_ghost and euclidean_dist_to_ghost <= 5:
            danger -= max(215 - 50 * euclidean_dist_to_ghost, 10)
    elif euclidean_dist_to_ghost <= 4:
        danger = 115 * (5.5 - euclidean_dist_to_ghost)
    return danger


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here, so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    return (
            currentGameState.getScore() +
            food_multiplier(currentGameState) -
            ghost_multiplier(currentGameState)
    )


# Abbreviation
better = betterEvaluationFunction
