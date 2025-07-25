import numpy as np
from rule_setting import default_board_state, mapper
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

    def getNextState(self, board, player, action):
        board = board.copy()
        x1, y1, x2, y2 = action
        p1 = board[y1, x1]
        p2 = board[y2, x2]
        if p1 * player <= 0:
            raise ValueError("非法操作：当前玩家不能移动该棋子")
        if p1 * p2 > 0:
            raise ValueError("非法操作：不能吃自己的子")
        board[y2, x2] = p1
        board[y1, x1] = 0
        return board, -player

    def getValidMoves(self, board, player):
        valid = np.zeros((10, 9, 10, 9), dtype=bool)
        no_eat_self = (board * player) <= 0
        for y in range(10):
            for x in range(9):
                tp = board[y, x]
                valid[y, x, :, :] = mapper[tp]['rule'](board, player, x, y, no_eat_self)

        for r1 in range(10):
            for c1 in range(9):
                piece = board[r1][c1]
                if piece * player <= 0:
                    continue
                for r2 in range(BOARD_ROWS):
                    for c2 in range(BOARD_COLS):
                        if self.is_legal_move(board, player, (r1, c1), (r2, c2)):
                            a = self.encode_action((r1, c1), (r2, c2))
                            valids[a] = 1
        return valids

    def getGameEnded(self, board, player):
        red_general = np.any(board == RED)
        black_general = np.any(board == BLACK)
        if red_general and black_general:
            return 0  # 未结束
        if red_general:
            return 1 if player == RED else -1
        if black_general:
            return 1 if player == BLACK else -1
        return 1e-4  # 平局或不明状态

    def getCanonicalForm(self, board, player):
        return board * player

    def stringRepresentation(self, board):
        return board.tostring()

    def is_legal_move(self, board, player, start, end):
        r1, c1 = start
        r2, c2 = end
        piece = board[r1][c1]
        target = board[r2][c2]
        if target * player > 0:
            return False  # 不能吃自己
        # 示例：只允许上下左右移动一步（将、卒）
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        if dr + dc != 1:
            return False
        return True

    def encode_action(self, start, end):
        s = start[0] * BOARD_COLS + start[1]
        e = end[0] * BOARD_COLS + end[1]
        return s * BOARD_ROWS * BOARD_COLS + e

    def decode_action(self, action):
        s_flat, e_flat = divmod(action, BOARD_ROWS * BOARD_COLS)
        return (s_flat // BOARD_COLS, s_flat % BOARD_COLS), (e_flat // BOARD_COLS, e_flat % BOARD_COLS)
