from typing import Deque
from template import Agent
import random
import time
from Reversi.reversi_model import ReversiGameRule


time_for_thinking = 0.95
important_untouchable_points = [(1,1), (1,6), (6,1), (6,6)]
disadvatage_point = [(1,0),(0,1),(0,6),(1,7),(6,7),(7,6),(6,0),(7,1)]
important_touchable_points = [(0,0),(0,7),(7,0),(7,7)]
advatage_point = [(2,0),(3,0),(4,0),(5,0),(2,7),(3,7),(4,7),(5,7),(0,2),(0,3),(0,4),(0,5),(7,2),(7,3),(7,4),(7,5)]

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = ReversiGameRule(2)
    
    def GetAction(self, game_state, agent_id):
        return self.game_rule.getLegalActions(game_state, agent_id)
    
    def Execute(self, game_state, action, agent_id):
        next_state = self.game_rule.generateSuccessor(game_state, action, agent_id)
        next_score = self.game_rule.calScore(next_state,agent_id)
        return (next_state, next_score)
    
    def OpponentGetAction(self, game_state, agent_id):
        return self.game_rule.getLegalActions(game_state, 1 - agent_id)

    def OpponentExecute(self, game_state, action, agent_id):
        next_state = self.game_rule.generateSuccessor(game_state, action, 1 - agent_id)
        next_score = self.game_rule.calScore(next_state, 1 - agent_id)
        return (next_state, next_score)

    def SelectAction(self,actions,game_state):
        self.game_rule.agent_colors = game_state.agent_colors
        count = 0
        Queue = Deque ([(game_state, [])])
        start_time = time.time()
        max_score = -1
        key_action_important = list(set(actions).intersection(set(important_touchable_points)))
        if len(key_action_important) == 1:
            return key_action_important[0]
        
        key_action_advatage = list(set(actions).intersection(set(advatage_point)))
        if len(key_action_advatage) == 1:
            return key_action_advatage[0]
            
        key_action_untouchable = list(set(actions).difference(set(important_untouchable_points),set(disadvatage_point)))
        if len(key_action_untouchable) > 0:
            actions = key_action_untouchable
        solve_method = random.choice(actions)
        while time.time()-start_time < time_for_thinking and len(Queue):
            count += 1
            game_state, path = Queue.popleft()
            new_action = self.GetAction(game_state, self.id)
            key_action_important = list(set(new_action).intersection(set(important_touchable_points)))
            if len(key_action_important) > 0:
                new_action = key_action_important
            key_action_advatage = list(set(new_action).intersection(set(advatage_point)))
            if len(key_action_advatage) > 0:
                new_action = key_action_advatage
            key_action_untouchable = list(set(new_action).difference(set(important_untouchable_points),set(disadvatage_point)))
            if len(key_action_untouchable) > 0:
                new_action = key_action_untouchable
            for action in new_action:
                if time.time()-start_time > time_for_thinking:
                    break
                next_path = path + [action]
                next_game_state, next_score = self.Execute(game_state, action, self.id)
                if self.GameEnd(next_game_state) and next_score > max_score:
                    max_score = next_score
                    solve_method = next_path[0]
                    break
                opponent_new_actions = self.OpponentGetAction(game_state, self.id)
                opponent_score = -1
                for opponent_action in opponent_new_actions:
                    next_game_state,opponent_next_score = self.OpponentExecute(next_game_state, opponent_action, self.id)
                    if opponent_score < opponent_next_score:
                        opponent_score = opponent_next_score
                    Queue.append((next_game_state, next_path))     
        return solve_method
    
    def GameEnd(self, state):
        return (self.GetAction(state,0) == ["Pass"] 
            and self.GetAction(state,1) == ["Pass"])
