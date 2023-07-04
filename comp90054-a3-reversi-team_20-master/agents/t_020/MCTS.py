from template import Agent
import random, time
from Reversi.reversi_model import ReversiGameRule
from collections import deque
from copy import deepcopy

think_time = 0.9
gamma = 0.85
epsilon = 0.7
highest_value_point = [(0, 0), (0, 7), (7, 0), (7, 7)]
high_value_point = [(2, 0), (0, 2), (7, 5), (5, 7), (0, 5), (5, 0), (7, 2), (2, 7)]
lowest_value_point = [(1, 1), (1, 6), (6, 1), (6, 6)]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.game_rule = ReversiGameRule(2)

    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    def GetOpponentActions(self, state):
        return self.game_rule.getLegalActions(state, 1 - self.id)

    def ExcuteAction(self, state, action):
        next_state = self.game_rule.generateSuccessor(state, action, self.id)
        next_score = self.game_rule.calScore(next_state, self.id)
        return (next_state, next_score)

    def ExcuteOpponentAction(self, state, action):
        next_state = self.game_rule.generateSuccessor(state, action, 1 - self.id)
        next_score = self.game_rule.calScore(next_state, 1 - self.id)
        return (next_state, next_score)

    def GameOver(self, state):
        return (self.GetActions(state) == ["Pass"] and self.GetOpponentActions(state) == ["Pass"])

    def CalculateScore(self, state):
        return self.game_rule.calScore(state, self.id) - self.game_rule.calScore(state, 1 - self.id)

    def SelectAction(self, actions, game_state):
        # init
        self.game_rule.agent_colors = game_state.agent_colors
        count = 0
        start_time = time.time()

        # Pre-processing
        alternative_actions = list((set(actions)) & (set(highest_value_point)))
        if len(alternative_actions) == 1:
            return alternative_actions[0]
        if len(alternative_actions) > 0:
            actions = alternative_actions
        else:
            alternative_actions = list((set(actions)) & (set(high_value_point)))
            if len(alternative_actions) == 1:
                return alternative_actions[0]
            if len(alternative_actions) > 0:
                actions = alternative_actions
            alternative_actions = list((set(actions)) - (set(lowest_value_point)))
            if len(alternative_actions) > 0:
                actions = alternative_actions
        solution = random.choice(actions)

        # MCT
        vs = dict()
        ns = dict()
        best_actionset = dict()
        expanded_actionset = dict()
        root_state = "r"
        def Expanded(state, actions):
            if state in expanded_actionset:
                expanded_actions = expanded_actionset[state]
                return list(set(actions).difference(set(expanded_actions)))
            else:
                return actions

        def OpponentMove(next_state):
            opponent_new_actions = self.GetOpponentActions(next_state)
            opponent_max_score = 0
            opponent_best_state = next_state
            for opponent_action in opponent_new_actions:
                opponent_next_state, opponent_next_score = self.ExcuteOpponentAction(next_state, opponent_action)
                if opponent_next_score > opponent_max_score:
                    opponent_max_score = opponent_next_score
                    opponent_best_state = opponent_next_state
                    opponent_best_action = opponent_action
            return opponent_best_state, opponent_best_action

        while time.time() - start_time < think_time:
            count += 1
            state = deepcopy(game_state)
            new_actions = actions
            search_state = root_state
            queue = deque([])
            reward = 0
            length = 0

            # Select
            while len(Expanded(search_state, new_actions)) == 0 and not self.GameOver(state):
                if time.time() - start_time >= think_time:
                    return solution
                if (random.uniform(0, 1) < epsilon) and (search_state in best_actionset):
                    search_action = best_actionset[search_state]
                else:
                    search_action = random.choice(new_actions)
                next_state, next_score = self.ExcuteAction(state, search_action)
                queue.append((search_state, search_action))
                opponent_best_state, opponent_best_action = OpponentMove(next_state)
                search_state = search_state + str(search_action[0]) + str(search_action[1]) + str(opponent_best_action[0]) + str(opponent_best_action[1])
                new_actions = self.GetActions(opponent_best_state)
                state = opponent_best_state

            # Expand
            available_actions = Expanded(search_state, new_actions)
            if len(available_actions) == 0:
                action = random.choice(new_actions)
            else:
                action = random.choice(available_actions)
            if search_state in expanded_actionset:
                expanded_actionset[search_state].append(action)
            else:
                expanded_actionset[search_state] = [action]
            queue.append((search_state, action))
            next_state, next_score = self.ExcuteAction(state, action)
            opponent_best_state, opponent_best_action = OpponentMove(next_state)
            search_state = search_state + str(action[0]) + str(action[1]) + str(opponent_best_action[0]) + str(opponent_best_action[1])
            new_actions = self.GetActions(opponent_best_state)
            state = opponent_best_state

            # Simulation
            while not self.GameOver(state):
                length += 1
                if time.time() - start_time >= think_time:
                    return solution
                search_action = random.choice(new_actions)
                next_state, next_score = self.ExcuteAction(state, search_action)
                opponent_best_state, opponent_best_action = OpponentMove(next_state)
                # iteration
                new_actions = self.GetActions(opponent_best_state)
                bonus_reward = list(set(new_actions).intersection(set(highest_value_point)))
                if len(bonus_reward) > 0:
                    reward += 10
                state = opponent_best_state
            reward += self.CalculateScore(state)


            # Backpropagete
            search_value = reward * (gamma ** length)
            while len(queue) and time.time() - start_time < think_time:
                mct_state, search_action = queue.pop()
                if mct_state in vs:
                    if search_value > vs[mct_state]:
                        vs[mct_state] = search_value
                        best_actionset[mct_state] = search_action
                    ns[mct_state] += 1
                else:
                    vs[mct_state] = search_value
                    ns[mct_state] = 1
                    best_actionset[mct_state] = search_action
                search_value *= gamma
            if root_state in best_actionset:
                solution = best_actionset[root_state]
        return solution