import json

from template import Agent
import random
import numpy as np
import pandas as pd
from Reversi.reversi_model import ReversiGameRule
from Reversi.reversi_utils import Cell
import time

epsilon = 0.9   # greedy
alpha = 0.1     # learning rate
gamma = 0.8     # Diminishing reward value
qValue = -1
# important position
fourCorner = [(0,0), (7,7), (0,7), (7,0)]
# unneccsary to occupied
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
        with open("agents/t_020/weight.json", "r", encoding="utf-8") as f:
            self.weight = json.load(f)["weight"]
        # print(self.weight)


    def GetScore(self, state):
        if self.rule.getLegalActions(state, self.id) == ["Pass"] \
        and self.rule.getLegalActions(state, 1 - self.id) == ["Pass"]:
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
                # player occupid corner and other side (around the board)

        # how many player on nearby corner - unnecssary points
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
        # feature 4 toal 4 means 4 four main corners
        features.append(feature3 / 4)
        # feature 4 toal 4 means 4 sub corners
        features.append(feature4 / 4)
        # feature 5 toal 24 means 24 edge point
        features.append(feature5 / 24)
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
        # player color
        self.rule.agent_colors = game_state.agent_colors
        startTime = time.time()
        result = random.choice(actions)
        qValue = -1
        for a in actions:
            newValue = self.QValue(game_state, a)
            if time.time() - startTime <= 0.8 and newValue > qValue:
                qValue = newValue
                result = a
        return result