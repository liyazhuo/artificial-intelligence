from sample_players import DataPlayer
import random
from math import sqrt, log

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation
    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.
    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least
        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.
        See RandomPlayer and GreedyPlayer in sample_players for more examples.
        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        #Select a random move at the beginning of a game
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:

            self.queue.put(random.choice(state.actions())) #call self.queue.put(ACTION) at least once          
            if not state.terminal_test():
                """Monte Carlo Tree Search """
                mcts = MonteCarloTreeSearch(GameTreeNode(state), e_e_ratio = 1.25)
                #e_e_ratio is the ratio between exploitation and exploration
                while True:
                    mcts.run_search()
                    best_action = mcts.best_action()
                    self.queue.put(best_action)

    #          """Alpha Beta Pruning with iterative deepening"""
    #          self.queue.put(self.alpha_beta_pruning(state, depth=7))    

    # def alpha_beta_pruning(self, state, depth):
        
    #     def min_value(state, alpha, beta, depth):
    #         if state.terminal_test(): return state.utility(self.player_id)
    #         if depth <= 0: return self.score(state)
    #         value = float("inf")
    #         for action in state.actions():
    #             value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
    #             if value <= alpha:
    #                 return value
    #             beta = min(beta, value)
    #         return value

    #     def max_value(state, alpha, beta, depth):
    #         if state.terminal_test(): return state.utility(self.player_id)
    #         if depth <= 0: return self.score(state)
    #         value = float("-inf")
    #         for action in state.actions():
    #             value = max(value, min_value(state.result(action),alpha, beta, depth - 1))
    #             if value >= beta:
    #                 return value
    #             alpha = max(alpha, value)
    #         return value
        
    #     alpha = float("-inf")
    #     beta = float("inf")
    #     return max(state.actions(), key=lambda x: min_value(state.result(x), alpha, beta,depth - 1))

    # def score(self, state):
    #     own_loc = state.locs[self.player_id]
    #     opp_loc = state.locs[1 - self.player_id]
    #     own_liberties = state.liberties(own_loc)
    #     opp_liberties = state.liberties(opp_loc)
    #     return len(own_liberties) - len(opp_liberties)

 
class GameTreeNode():
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.num_visited = 0
        self.reward = 0
        self.untried_actions = state.actions()
    
    def Expand(self): #Node expansion
        next_action = self.untried_actions.pop(random.randint(0, len(self.untried_actions)-1))
        new_child = GameTreeNode(self.state.result(next_action), parent=self)
        self.children[new_child] = next_action 
        #Here children dictionary stores all the actions while keys are child nodes
        return new_child

    def best_uct(self, e_e_ratio):
        #UCT algorithm for node selections
        best_child = max(self.children.keys(), key = lambda child: child.reward/child.num_visited + e_e_ratio * sqrt(log(self.num_visited)/child.num_visited))
        return best_child


    def is_terminal(self):
        return self.state.terminal_test()

    def best_action(self):
        #action associated with the best node
        #Here the mosted visited nodes are selected
        best_child = max(self.children.keys(), key = lambda c: c.num_visited)
        return self.children[best_child]



class MonteCarloTreeSearch():

    def __init__(self, node, e_e_ratio):
        self.root = node
        self.e_e_ratio = e_e_ratio


    def tree_policy(self): #Tree policy for expansion using UCT
        current_node = self.root
        while not current_node.is_terminal() :
            if current_node.untried_actions:
                return current_node.Expand()
            else:
                current_node = current_node.best_uct(self.e_e_ratio)
        return current_node
    


    def default_policy(self, state):# the default policy used to simulate 
        while not state.terminal_test():
            state = state.result(random.choice(state.actions())) #uniformly random moves
        return self.state_reward(state)

    def state_reward(self, state): #reward policy
        if state.utility(state.player()) == float("inf"):
            return -1.0
        elif state.utility(state.player()) == float("-inf"):
            return 1.0
        else:
            return 0.0


    def back_up_negameax(self, node, delta): # backpropagations after simulation
            node.num_visited = node.num_visited + 1
            node.reward = node.reward + delta
            if node.parent:
                self.back_up_negameax(node.parent, -delta)

    def run_search(self):
        start_node = self.tree_policy()
        delta = self.default_policy(start_node.state)
        self.back_up_negameax(start_node, delta)

    def best_action(self):
        return self.root.best_action()