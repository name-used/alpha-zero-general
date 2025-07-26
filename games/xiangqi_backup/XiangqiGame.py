import numpy as np
from .rule_setting import default_board_state, mapper
from Game import Game


class XiangqiGame(Game):
    def __init__(self):
        super(XiangqiGame, self).__init__()

    def getInitBoard(self):
        return default_board_state.copy()

    def getBoardSize(self):
        return 10, 9

    def getActionSize(self):
        return 8100

    def getInputChannels(self):
        return 15

    def getNextState(self, board, player, action):
        board = board.copy()
        x1, y1, x2, y2 = self.action2position(action)
        p1 = board[y1, x1]
        p2 = board[y2, x2]
        if p1 * player <= 0:
            raise ValueError("非法操作：当前玩家不能移动该棋子")
        if p1 * p2 > 0:
            raise ValueError("非法操作：不能吃自己的子")
        valid = mapper[p1]['rule'](board, player, x1, y1, np.ones_like(board))
        if not valid[y2, x2]:
            raise ValueError("非法操作：走子规则不合法")
        board[y2, x2] = p1
        board[y1, x1] = 0
        return board, -player

    def getValidMoves(self, board, player):
        valid = np.zeros((10, 9, 10, 9), dtype=bool)
        no_eat_self = (board * player) <= 0
        for y in range(10):
            for x in range(9):
                tp = board[y, x]
                if tp * player > 0:
                    valid[y, x, :, :] = mapper[tp]['rule'](board, player, x, y, no_eat_self)
        return valid.flatten()

    def getGameEnded(self, board, player):
        red_general = (board == 1).any()
        black_general = (board == -1).any()
        if red_general and black_general:
            return 0  # 未结束
        if red_general:
            return 1 if player == 1 else -1
        if black_general:
            return 1 if player == -1 else -1
        raise 1111111111  # 未知状态

    def getCanonicalForm(self, board, player):
        if player > 0: return board
        return np.rot90(board * -1, 2)

    def stringRepresentation(self, board):
        return board.tostring()

    def action2position(self, action):
        xy1, xy2 = action // 90, action % 90
        y1, x1 = xy1 // 9, xy1 % 9
        y2, x2 = xy2 // 9, xy2 % 9
        return x1, y1, x2, y2

    def getSymmetries(self, board, pi):
        return [(board, pi)]
        # 原样
        symmetries = [(board, pi)]
        # 镜像棋盘
        flipped_board = np.fliplr(board)
        # pi shape: (10, 9, 10, 9)
        pi_board = np.asarray(pi).reshape((10, 9, 10, 9))
        # 镜像：x1, x2 对称映射 → flip axis=1 和 axis=3
        flipped_pi_board = np.flip(pi_board, axis=(1, 3))
        symmetries.append((flipped_board, flipped_pi_board.flatten().tolist()))
        return symmetries
