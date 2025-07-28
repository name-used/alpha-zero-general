import torch
import logging
import os
from collections import deque
from pickle import Pickler
from random import shuffle
import numpy as np
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS
from games.xiangqi_jassor.model import Model
from games.xiangqi_jassor.state import GameState

log = logging.getLogger(__name__)


class Coach:
    def __init__(self, model: Model, args, mcts_args, arena_args):
        self.model = model
        self.args = args
        self.mcts_args = mcts_args
        self.arena_args = arena_args
        self.trainExamplesHistory = []

    def learn(self):
        best_model = Model(None)
        for i in range(1, self.args.epoch + 1):
            # 自对弈搜索，这一部分可以用皮卡鱼引擎替代
            # self_play_times 即每 epoch 的对弈搜索次数，搜索算法使用的是蒙特卡罗
            # 这个过程的作用是“生成数据集”，为对弈环节形成的每个局面都进行一次评分
            log.info(f'Starting Iter #{i} ...')
            iterationTrainExamples = deque([], maxlen=self.args.history_limit)
            for _ in tqdm(range(self.args.self_play_times), desc="Self Play"):
                # mcts = MCTS(self.model, self.mcts_args)  # reset search tree
                mcts = MCTS(None, self.mcts_args)  # reset search tree
                iterationTrainExamples += self.executeEpisode(mcts)

            # 自对弈结果管控 + 制作数据集
            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.history_epoch_limit:
                log.warning(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            # self.saveTrainExamples(i - 1)
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            pmcts = MCTS(best_model, self.mcts_args)
            self.model.train(trainExamples)
            nmcts = MCTS(self.model, self.mcts_args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.arena_args.compare_times)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.replace_if:
                log.info('REJECTING NEW MODEL')
                self.model.net.load_state_dict(best_model.net.state_dict())
            else:
                log.info('ACCEPTING NEW MODEL')
                best_model.net.load_state_dict(self.model.net.state_dict())
                torch.save(self.model.net.state_dict(), self.args.checkpoint / f'{i}.pth')
                torch.save(self.model.net.state_dict(), self.args.checkpoint / f'best.pth')

    def executeEpisode(self, mcts):
        trainExamples = []
        state = GameState()
        episodeStep = 0
        while True:
            episodeStep += 1
            temperature = int(episodeStep < self.args.explore_num)

            pi = mcts.getActionProb(state, temperature=temperature)
            trainExamples.append([state.clone(), pi, 0])
            # sym 是对称性部分的代码，先不管，后面再说
            # sym = self.game.getSymmetries(state, pi)
            # for b, p in sym:
            #     trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            state.do_action(action)

            # 嗯……此处循环判别的是“对局是否结束”，一次对局要走到结束才算完，当然，也可以用 max_episode 强行规定平局
            r = state.end_state()
            if r != 0:
                for item in trainExamples:
                    item[2] = r
                return trainExamples
            # 达到上界，强制平局
            if 0 < self.args.max_episode <= episodeStep:
                print(f"[Draw] Max steps reached: {episodeStep}")
                return trainExamples  # 视为和棋

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f'{iteration}.examples')
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
