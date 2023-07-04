import imp
import json
import time

from template import Agent
import random
import numpy as np
import pandas as pd
from Reversi.reversi_model import ReversiGameRule
from Reversi.reversi_utils import Cell
import agents.t_020.myTeam
import time


epsilon = 0.9   # greedy
alpha = 0.1     # learning rate
gamma = 0.8     # Diminishing reward value
qValue = -1
fourCorner = [(0,0), (7,7), (0,7), (7,0)]
subCorner = [(1,1), (1,6), (6,1), (6,6)]
edge = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(2,0), \
        (3,0),(4,0),(5,0),(6,0), (7,1), (7,2), (7,3),\
        (7,4),(7,5), (7,6), (1,7), (2,7),(3,7), (4,7), (5,7),(6,7)]


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.rule = ReversiGameRule(2)
        # initial weight
        self.weight = [0,0,0,0,0,0]
        # print(self.weight)

    def GetScore(self, state):
        if self.rule.getLegalActions(state, self.id) == ["Pass"] \
        and self.rule.getLegalActions(state, 1-self.id) == ["Pass"]:
            return self.rule.calScore(state, self.id) - self.rule.calScore(state, 1 - self.id)
        else:
            return 0

    def Feature(self, state, action):
        features = []
        # how many player's color on the board - player's color - total is 64
        feature1 = self.rule.calScore(state, self.id)/64
        # how many competitor's color on the board - competitors color
        feature2 = self.rule.calScore(state, 1 - self.id)/64
        # how many player on the four corner
        next_state = self.rule.generateSuccessor(state, action, self.id)
        feature3 = 0
        feature4 = 0
        feature5 = 0

        for i in range(len(fourCorner)):
            if next_state.board[fourCorner[i][0]][fourCorner[i][1]] == self.rule. \
                agent_colors[self.id]:
                feature3 += 1

        # how many player on nearby corner
        for i in range(len(subCorner)):
            if next_state.board[fourCorner[i][0]][fourCorner[i][1]] == Cell.EMPTY \
                    and next_state.board[subCorner[i][0]][subCorner[i][1]] == self.rule. \
                agent_colors[self.id]:
                feature4 += 1

        # player occupid edge point on the board
        for i in range(len(edge)):
            if next_state.board[edge[i][0]][edge[i][1]] == self.rule. \
                agent_colors[self.id]:
                feature5 += 1

        features.append(feature1)
        features.append(feature2)
        features.append(feature3 / 4)
        features.append(feature4 / 4)
        features.append(feature5 / 26)
        return features

    def QValue(self,state, action):
        if self.Feature(state, action) is not None:
            # initial q value
            return qValue
        else:
            for i in range(5):
                value += self.Feature(state, action)[i] * self.weight[i]
            return value


    def SelectAction(self,actions,game_state):
        # match our agent color
        self.rule.agent_colors = game_state.agent_colors
        startTime = time.time()
        result = random.choice(actions)
        best_Q_value = -1
        if random.uniform(0,1) < 1 - epsilon:
            for a in actions:
                if time.time() - startTime <= 0.9:
                    value = self.QValue(game_state, a)
                if value > best_Q_value:
                    best_Q_value = value
                    result = a
        else:
            value = self.QValue(game_state, result)
            best_Q_value = value
        next_state = self.rule.generateSuccessor(game_state, result, self.id)
        C_action = self.rule.getLegalActions(next_state, 1 - self.id)
        CScore = 0
        C_best = next_state
        for C in C_action:
            C_next = self.rule.generateSuccessor(next_state, C, 1-self.id)
            CN_score = self.rule.calScore(C_next, 1-self.id)
            if CN_score > CScore:
                CScore = CN_score
                C_best = C_next
        next_state = C_best
        reward = self.GetScore(next_state)

        next_actions = self.rule.getLegalActions(next_state, self.id)
        best_next_Q_value = -1
        for a in actions:
            if time.time() - startTime <= 0.9:
                value = self.QValue(next_state, a)
                best_next_Q_value = max(best_Q_value, value)
        features = self.Feature(game_state, result)
        delta = reward + gamma * best_next_Q_value - best_Q_value
        for i in range(len(features)):
            self.weight[i] += alpha * delta * features[i]
        with open("agents/t_020/weight.json", "w", encoding="utf-8") as fw:
            json.dump({"weight": self.weight}, fw, indent=4, ensure_ascii=False)
        return result
    def GetWeight(self):
        return self.weight