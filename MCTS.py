import logging
import math
from games.xiangqi_jassor.state import GameState
import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, state: GameState, temperature=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(state.clone(), self.args.search_limit)

        s = state.keystr()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(32400)]
        counts = np.asarray(counts) + state.getValidMoves() * 1e-19

        if temperature == 0:
            best_action = counts.argmax()
            probs = np.zeros_like(counts)
            probs[best_action] = 1.0
            return probs

        counts = np.asarray(counts) ** (1. / temperature)
        assert counts.max() > counts.min() >= 0
        return counts / counts.sum()
        # counts = [x ** (1. / temperature) for x in counts]
        # counts_sum = float(sum(counts))
        # probs = [x / counts_sum for x in counts]
        # return probs

    def search(self, state: GameState, max_search=200):
        if max_search <= 0:
            return 0

        s = state.keystr()
        if s not in self.Es:
            self.Es[s] = state.end_state()
        if self.Es[s] != 0:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            # 训练早期不使用 model 时，直接用 score 评分，后期用 model 了就 model 评分
            valids = state.getValidMoves()
            if self.model:
                self.Ps[s], v = self.model.predict(state)
                self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            else:
                # score 是 [-1, 1]，概率是 [0, 1]
                ps = np.asarray([(state.clone().do_action(a).score() + 1.001) ** 2 if v else 0 for a, v in enumerate(valids)])
                ps = ps / ps.sum()
                v = np.mean(ps)
                self.Ps[s] = ps
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(32400):
            if not valids[a]: continue
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
            u = u * state.player_turn
            if u > cur_best:
                cur_best = u
                best_act = a

        if best_act == -1:
            # 没有合法动作
            raise 'no valid method'
        a = best_act
        state.do_action(a)

        v = self.search(state, max_search-1)
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return v
