import logging
from types import SimpleNamespace
import coloredlogs
from pathlib import Path
from Coach import Coach
from games.xiangqi_jassor.model import Model

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = SimpleNamespace(
    continue_checkpoint=None,
    coach=SimpleNamespace(
        checkpoint_root=Path('./temp/'),
        epoch=10,               # 训练执行的轮次数
        self_play_times=100,    # 每轮训练生成多少次探索对局
        explore_num=20,         # 每次探索中前多少步使用随机策略探索（之后固定使用最大概率探索）
        replace_if=0.5,         # 如果新模型对旧模型胜率达到这个，则保留（此逻辑待商议，可能要整体移除）
        history_limit=200_000,  # 最多保持多少个探索对局（旧策略局面数的移除限制）
        history_epoch_limit=20, # 最多保持多少个探索对局（旧策略epoch数的移除限制）
        max_episode=-1,         # 最大对弈轮次数
        hard_score_epoch=3,     # 使用盘面评分的轮次数
    ),
    mcts=SimpleNamespace(
        numMCTSSims=100,        # 每次探索对局中的每一步执行多少次搜索（生成 UCB 评分）
        search_limit=20,        # 单次搜索的深度（这个是我的自定义限制，实际上可以取消掉，粮食耗尽自然停止）
        cpuct=1,                # 探索/利用平衡参数
    ),
    arena=SimpleNamespace(
        compare_times=40,       # 模型筛选评估时
    ),
)


def main():
    model = Model(batch_size=16)
    if args.continue_checkpoint:
        model.net.load_state_dict(args.continue_checkpoint)
    c = Coach(model, args.coach, args.mcts, args.arena)
    c.learn()


if __name__ == "__main__":
    main()
