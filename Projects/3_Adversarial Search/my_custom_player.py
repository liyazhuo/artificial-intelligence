
from sample_players import DataPlayer
import random
from math import sqrt, log

SCALER_FACTOR = 1.0/sqrt(2.0)

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
        #import random
        if state.ply_count < 2:            
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(random.choice(state.actions()))
            if not state.terminal_test():
                mcts = MonteCarloTreeSearch(state, SCALER_FACTOR)
                mcts.run_search()
                self.queue.put(mcts.best_actions())

class GameTreeNode():
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent
        self.children = []
        self.num_visited = 0
        self.reward = 0
        self.untried_actions = state.actions()
    
    def Expand(self):
        a = random.choice(self.untried_actions())
        new_child = GameTreeNode(self.state.result(a), parent=self)
        new_child.children = a
        return new_child
    
    def best_child(self, scale_factor):
        f = lambda child : child.reward/child.num_visited + scale_factor*sqrt(2.0*log(self.num_visited)/child.num_visited)
        bc = max(self.children, key = f)
        return bc 

    def is_expandable(self):
        if not self.is_terminal() and self.children :
            return True
        else:
            return False

    def is_terminal(self):
        return self.state.terminal_test()

 



class MonteCarloTreeSearch():

    def __init__(self, state, scale_factor):
        self.node = GameTreeNode(state)
        self.scale_factor = scale_factor


    def tree_policy(self, node):
        while not node.is_terminal :
            if node.is_expandable():
                return node.Expand()
            else:
                node = node.best_child(SCALER_FACTOR)
                print (node)
        return node

    def default_policy(self, state):
        while not state.terminal_test():
            a = random.choice(state.actions())
            new_state = state.result(a)
        return self.state_reward(new_state)

    def state_reward(self, state):
        if state.utility(state.player()) == float("inf"):
            return -1.0
        elif state.utility(state.player()) == float("-inf"):
            return 1.0
        else:
            return 0

    def back_up_negameax(self, node, delta):
        while node.parent :
            node.num_visited = node.num_visited + 1
            node.reward = node.reward + delta
            delta = -delta
            node = node.parent 

    def run_search(self):
        start_node = self.tree_policy(self.node)
        start_state = start_node.state
        delta = self.default_policy(start_state)
        self.back_up_negameax(start_node, delta)

    def best_actions(self):
        return self.node.best_child

