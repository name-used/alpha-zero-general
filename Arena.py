import logging
import time

from tqdm import tqdm
from games.xiangqi_jassor.state import GameState
from games.xiangqi_jassor.ui import UI

log = logging.getLogger(__name__)


class Arena:
    def __init__(self, player1, player2, display):
        self.player1 = player1
        self.player2 = player2
        self.ui = UI() if display else None

    def playGame(self):
        state = GameState()
        if self.ui: self.ui.display(state)
        while state.end_state() == 0:
            action = self.player1(state)
            state.do_action(action)
            if self.ui: self.ui.flush()
            time.sleep(1)
            action = self.player2(state)
            state.do_action(action)
            if self.ui: self.ui.flush()
            time.sleep(1)
        if self.ui: self.ui.end()
        return state.end_state()

    def playGames(self, num):
        num = num // 2
        s1 = [self.playGame() for _ in tqdm(range(num), desc="Arena.playGames player1-red")]
        self.player1, self.player2 = self.player2, self.player1
        s2 = [self.playGame() for _ in tqdm(range(num), desc="Arena.playGames player1-black")]
        ss = [len([s for s in s1+s2 if s == end_state]) for end_state in [1, -1, 0]]
        return ss
