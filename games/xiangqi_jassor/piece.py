import numpy as np

mk = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ms = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
mm = [(-2, 1), (2, 1), (-2, -1), (2, -2), (-1, 2), (1, 2), (-1, -2), (1, -2)]


def an(state, x, y):
    return np.zeros_like(state.board, dtype=bool)


def ak(state, x, y):
    legal = np.zeros_like(state.board, dtype=bool)
    alert = state.red_alert if state.player_turn > 0 else state.black_alert
    # 行动表
    ly1, ly2 = (7, 10) if state.player_turn > 0 else (0, 3)
    for p, q in mk:
        nx = x + p
        ny = y + q
        # 走子规则
        if alert >= 1 and 3 <= nx < 6 and ly1 <= ny < ly2 and state.board[ny, nx].item() == 0:
            legal[ny, nx] = True
        # 吃子规则
        if alert >= 2 and 3 <= nx < 6 and ly1 <= ny < ly2 and state.board[ny, nx].item() * state.player_turn < 0:
            legal[ny, nx] = True
    # 特殊吃子，对脸吃王
    if alert == 3:
        line = state.board[:, x]
        if -1 in line:
            y2 = np.argwhere(line == -1).item()
            p = min(y, y2)
            q = max(y, y2)
            if not line[p+1: q].any():
                legal[y2, x] = True
    # 特殊走子，允许停招(红棋必须有停招许可)
    if state.player_turn < 0 or state.red_stop_limit > 0:
        legal[y, x] = True
    return legal


def a_s(state, x, y):
    legal = np.zeros_like(state.board, dtype=bool)
    alert = state.red_alert if state.player_turn > 0 else state.black_alert
    # 行动表
    ly1, ly2 = (7, 10) if state.player_turn > 0 else (0, 3)
    for p, q in ms:
        nx = x + p
        ny = y + q
        # 走子规则
        if alert >= 1 and 3 <= nx < 6 and ly1 <= ny < ly2 and state.board[ny, nx].item() == 0:
            legal[ny, nx] = True
        # 吃子规则
        if alert >= 2 and 3 <= nx < 6 and ly1 <= ny < ly2 and state.board[ny, nx].item() * state.player_turn < 0:
            legal[ny, nx] = True
    return legal


def ax(state, x, y):
    legal = np.zeros_like(state.board, dtype=bool)
    alert = state.red_alert if state.player_turn > 0 else state.black_alert
    # 行动表
    ly1, ly2 = (5, 10) if state.player_turn > 0 else (0, 5)
    for p, q in ms:
        nx = x + p * 2
        ny = y + q * 2
        # 塞象眼
        if 0 <= nx < 9 and ly1 <= ny < ly2:
            tx = x + p
            ty = y + q
            if not state.board[ty, tx].item():
                # 走子规则
                if alert >= 1 and state.board[ny, nx].item() == 0:
                    legal[ny, nx] = True
                # 吃子规则
                if alert >= 2 and state.board[ny, nx].item() * state.player_turn < 0:
                    legal[ny, nx] = True
    return legal


def am(state, x, y):
    legal = np.zeros_like(state.board, dtype=bool)
    alert = state.red_alert if state.player_turn > 0 else state.black_alert
    # 行动表
    for p, q in mm:
        nx = x + p
        ny = y + q
        # 别马腿
        if 0 <= nx < 9 and 0 <= ny < 10:
            tx = x + int(p / 2)
            ty = y + int(q / 2)
            if not state.board[ty, tx].item():
                # 走子规则
                if alert >= 1 and state.board[ny, nx].item() * state.player_turn == 0:
                    legal[ny, nx] = True
                # 吃子规则
                if alert >= 2 and state.board[ny, nx].item() * state.player_turn < -1:
                    legal[ny, nx] = True
                # 吃王规则
                if alert == 3 and state.board[ny, nx].item() * state.player_turn == -1:
                    legal[ny, nx] = True
    return legal


def ac(state, x, y):
    legal = np.zeros_like(state.board, dtype=bool)
    alert = state.red_alert if state.player_turn > 0 else state.black_alert
    # 行动表
    for p, q in mk:
        for r in range(1, 10):
            nx = x + p * r
            ny = y + q * r
            if 0 <= nx < 9 and 0 <= ny < 10:
                # 走子规则
                if state.board[ny, nx].item() * state.player_turn == 0:
                    if alert >= 1:
                        legal[ny, nx] = True
                    continue
                # 吃子规则
                if alert >= 2 and state.board[ny, nx].item() * state.player_turn < -1:
                    legal[ny, nx] = True
                # 吃王规则
                if alert == 3 and state.board[ny, nx].item() * state.player_turn == -1:
                    legal[ny, nx] = True
                break
            else:
                break
    return legal


def ap(state, x, y):
    legal = np.zeros_like(state.board, dtype=bool)
    alert = state.red_alert if state.player_turn > 0 else state.black_alert
    # 行动表
    for p, q in mk:
        attack = False
        for r in range(1, 10):
            nx = x + p * r
            ny = y + q * r
            if 0 <= nx < 9 and 0 <= ny < 10:
                if not attack:
                    if state.board[ny, nx].item():
                        attack = True
                    else:
                        # 走子规则
                        if alert >= 1:
                            legal[ny, nx] = True
                else:
                    if state.board[ny, nx].item():
                        # 吃子规则
                        if alert >= 2 and state.board[ny, nx].item() * state.player_turn < -1:
                            legal[ny, nx] = True
                        # 吃王规则
                        if alert == 3 and state.board[ny, nx].item() * state.player_turn == -1:
                            legal[ny, nx] = True
                        break
            else:
                break
    return legal


def ab(state, x, y):
    legal = np.zeros_like(state.board, dtype=bool)
    alert = state.red_alert if state.player_turn > 0 else state.black_alert
    # 行动表
    for p, q in mk:
        if q * state.player_turn == 1: continue
        if (y-5) * state.player_turn > 0 and q == 0: continue
        nx = x + p
        ny = y + q
        if 0 <= nx < 9 and 0 <= ny < 10:
            # 走子规则
            if alert >= 1 and state.board[ny, nx].item() * state.player_turn == 0:
                legal[ny, nx] = True
            # 吃子规则
            if alert >= 2 and state.board[ny, nx].item() * state.player_turn < -1:
                legal[ny, nx] = True
            # 吃王规则
            if alert == 3 and state.board[ny, nx].item() * state.player_turn == -1:
                legal[ny, nx] = True
    return legal
