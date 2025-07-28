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
            self.search(state, self.args.search_limit)

        s = state.keystr()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(32400)]

        if temperature == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temperature) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

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
                self.Ps[s] = np.asarray([state.clone().do_action().score() * v for a, v in enumerate(valids)])
                v = np.mean(self.Ps[s])

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

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
