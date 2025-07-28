import numpy as np
import base64
from .piece import an, ak, a_s, ax, am, ac, ap, ab

# 0: 停招 1: 保子 2: 叫吃 3: 将军
a_r = [5, -2, -3, -4]
# 2、3、7: 士象兵各一分, 4、6: 马炮三分, 5: 车五分
s_t = np.asarray([0, 0, 1, 1, 3, 5, 3, 1])

# 棋盘定义
default_board = np.asarray([
    [-5, -4, -3, -2, -1, -2, -3, -4, -5],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0, -6,  0,  0,  0,  0,  0, -6,  0],
    [-7,  0, -7,  0, -7,  0, -7,  0, -7],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 7,  0,  7,  0,  7,  0,  7,  0,  7],
    [ 0,  6,  0,  0,  0,  0,  0,  6,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 5,  4,  3,  2,  1,  2,  3,  4,  5],
], dtype=int)
mapper = {tp: {'name': name, 'key': key, 'rule': rule} for tp, name, key, rule in zip(
    [0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7],
    ['－', '帅', '仕', '相', '码', '車', '砲', '兵', '将', '士', '象', '马', '车', '炮', '卒'],
    ['n', 'K', 'S', 'X', 'M', 'C', 'P', 'B', 'k', 's', 'x', 'm', 'c', 'p', 'b'],
    [an, ak, a_s, ax, am, ac, ap, ab, ak, a_s, ax, am, ac, ap, ab],
)}


class GameState:
    def __init__(self):
        self.board = default_board.copy()
        self.red_rations = 120
        self.black_rations = 100
        self.red_alert = 4
        self.black_alert = 4
        self.player_turn = 1
        self.red_stop_limit = 6

    def clone(self):
        state = GameState()
        state.board = self.board.copy()
        state.red_rations = self.red_rations
        state.black_rations = self.black_rations
        state.red_alert = self.red_alert
        state.black_alert = self.black_alert
        state.player_turn = self.player_turn
        state.red_stop_limit = self.red_stop_limit
        return state

    def do_action(self, action):
        a, p = action % 4, action // 4
        xy1, xy2 = p // 90, p % 90
        y1, x1 = xy1 // 9, xy1 % 9
        y2, x2 = xy2 // 9, xy2 % 9
        p1 = self.board[y1, x1].item()
        p2 = self.board[y2, x2].item()
        if p1 == p2 == self.player_turn:
            if self.player_turn > 0 and self.red_stop_limit <= 0:
                raise ValueError("非法操作：红棋停招次数已达上限")
            # 特殊规则，王走原位 = 停招
            if self.player_turn == 1: self.red_alert = a
            else: self.black_alert = a
            if self.player_turn == 1: self.red_rations += a_r[a]
            else: self.black_rations += a_r[a]
            if self.player_turn == 1: self.red_stop_limit -= 1
            self.player_turn = -self.player_turn
            return self
        if (self.red_rations if self.player_turn > 0 else self.black_rations) + a_r[a] < 0:
            raise ValueError(f"非法操作：军粮不足，无法执行当前声明 {a}")
        if p1 * self.player_turn <= 0:
            raise ValueError("非法操作：当前玩家不能移动该棋子")
        if p2 * self.player_turn > 0:
            raise ValueError("非法操作：不能吃自己的子")
        valid = mapper[abs(p1)]['rule'](self, x1, y1)
        if not valid[y2, x2]:
            raise ValueError("非法操作：走子规则不合法")
        # 走子
        self.board[y2, x2] = p1
        self.board[y1, x1] = 0
        # 改变状态
        if self.player_turn == 1: self.red_alert = a
        else: self.black_alert = a
        if self.player_turn == 1: self.red_rations += a_r[a]
        else: self.black_rations += a_r[a]
        self.player_turn = -self.player_turn
        return self

    def getValidMoves(self):
        # 上一步的走子声明（决定这一回合可以怎么走）
        valid = np.zeros((10, 9, 10, 9, 4), dtype=bool)
        for y in range(10):
            for x in range(9):
                # 拿的必须是自己的棋子
                tp = self.board[y, x]
                if tp * self.player_turn > 0:
                    # 无论下一回合的走子声明是什么，都不影响本回合的可行招法
                    # 但下一回合的声明招法与本回合军粮剩余有关 —— 2军粮刚好支持招法 0、1, 3 军粮刚好支持招法 0、1、2, 4 军粮刚好支持招法 0、1、2、3
                    m = max(1, min(4, self.red_rations if self.player_turn > 0 else self.black_rations))
                    valid[y, x, :, :, :m] = mapper[abs(tp)]['rule'](self, x, y)[..., None]
        return valid.flatten()

    def end_state(self):
        # 红棋战败规则：要么军粮耗尽无法再停招，要么老将被吃
        if self.red_stop_limit <= 0 and self.red_alert == 0: return -1
        if 1 not in self.board: return -1
        # 黑棋战败规则：老将被吃
        if -1 not in self.board: return 1
        # 其余状态：未结束
        return 0

    def keystr(self):
        board = (self.board + 8).astype(np.uint8)
        board = (board[:5] << 4) | board[5:]
        board_bytes = board.astype(np.uint8).tobytes()  # 快速压缩成稳定字节串
        board_str = base64.b64encode(board_bytes).decode('ascii')  # 转为 str
        key_parts = [
            str(self.red_rations),
            str(self.black_rations),
            str(self.red_alert),
            str(self.black_alert),
            str(self.player_turn),
            str(self.red_stop_limit),
            board_str,
        ]
        return '|'.join(key_parts)

    def score(self):
        # 红方视角的评分
        # 如果把老将吃掉了，就返回胜负评分
        if 1 not in self.board: return -1.0
        if -1 not in self.board: return 1.0
        # 否则按子力价值 * 剩余军粮评分
        red_score = s_t[self.board[self.board > 0]].sum()
        black_score = s_t[self.board[self.board < 0] * -1].sum()
        score = red_score * self.red_rations - black_score * self.black_rations
        return score / 10_000
