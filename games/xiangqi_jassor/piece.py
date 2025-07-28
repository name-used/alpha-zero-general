import numpy as np

mk = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ms = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
mm = [(-2, 1), (2, 1), (-2, -1), (2, -2), (-1, 2), (1, 2), (-1, -2), (1, -2)]


def an(board, player, x, y):
    return np.zeros_like(board, dtype=bool)


def ak(board, player, x, y):
    legal = np.zeros_like(board, dtype=bool)
    # 行动表
    for p, q in mk:
        nx = x + p
        ny = y + q
        # 九宫格限制
        if 3 <= nx < 6 and 7 <= ny < 10 and board[ny, nx].item() <= 0:
            legal[ny, nx] = True
    # 特殊吃子，对脸吃王
    line = board[:, x]
    if -1 in line:
        y2 = np.argwhere(line == -1).item()
        p = min(y, y2)
        q = max(y, y2)
        if not line[p+1: q].any():
            legal[y2, x] = True
    return legal


def a_s(board, player, x, y):
    legal = np.zeros_like(board, dtype=bool)
    # 行动表
    for p, q in ms:
        nx = x + p
        ny = y + q
        if 3 <= nx < 6 and 7 <= ny < 10 and board[ny, nx].item() <= 0:
            legal[ny, nx] = True
    return legal


def ax(board, player, x, y):
    legal = np.zeros_like(board, dtype=bool)
    # 行动表
    for p, q in ms:
        nx = x + p * 2
        ny = y + q * 2
        # 塞象眼
        if 0 <= nx < 9 and 6 <= ny < 10:
            tx = x + p
            ty = y + q
            if not board[ty, tx].item() and board[ny, nx].item() <= 0:
                legal[ny, nx] = True
    return legal


def am(board, player, x, y):
    legal = np.zeros_like(board, dtype=bool)
    # 行动表
    for p, q in mm:
        nx = x + p
        ny = y + q
        # 别马腿
        if 0 <= nx < 9 and 0 <= ny < 10:
            tx = x + int(p / 2)
            ty = y + int(q / 2)
            if not board[ty, tx].item() and board[ny, nx].item() <= 0:
                legal[ny, nx] = True
    return legal


def ac(board, player, x, y):
    legal = np.zeros_like(board, dtype=bool)
    # 行动表
    for p, q in mk:
        for r in range(1, 10):
            nx = x + p * r
            ny = y + q * r
            if 0 <= nx < 9 and 0 <= ny < 10 and board[ny, nx].item() <= 0:
                legal[ny, nx] = True
                if board[ny, nx].item():
                    break
            else:
                break
    return legal


def ap(board, player, x, y):
    legal = np.zeros_like(board, dtype=bool)
    # 行动表
    for p, q in mk:
        attack = False
        for r in range(1, 10):
            nx = x + p * r
            ny = y + q * r
            if 0 <= nx < 9 and 0 <= ny < 10:
                if not attack:
                    if board[ny, nx].item():
                        attack = True
                    else:
                        legal[ny, nx] = True
                else:
                    if board[ny, nx].item():
                        if board[ny, nx].item() <= 0:
                            legal[ny, nx] = True
                        break
            else:
                break
    return legal


def ab(board, player, x, y):
    legal = np.zeros_like(board, dtype=bool)
    # 行动表
    for p, q in mk:
        if q == 1: continue
        if y >= 5 and q != -1: continue
        nx = x + p
        ny = y + q
        if 0 <= nx < 9 and 0 <= ny < 10 and board[ny, nx].item() <= 0:
            legal[ny, nx] = True
    return legal
