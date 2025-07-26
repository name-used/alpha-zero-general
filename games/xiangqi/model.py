import torch
import torch.nn.functional as F
import numpy as np


class AlphaZeroXiangqiNet(torch.nn.Module):
    def __init__(self, board_planes=15, action_size=2086, num_blocks=10):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(board_planes, 256, kernel_size=(3, 3), padding=1)
        self.bn_in = torch.nn.BatchNorm2d(256)

        self.res_blocks = torch.nn.Sequential(*[ResidualBlock(256) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
        self.policy_bn = torch.nn.BatchNorm2d(2)
        self.policy_fc = torch.nn.Linear(2 * 10 * 9, action_size)

        # Value head
        self.value_conv = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
        self.value_bn = torch.nn.BatchNorm2d(1)
        self.value_fc1 = torch.nn.Linear(10 * 9, 256)
        self.value_fc2 = torch.nn.Linear(256, 1)

    def forward(self, x):  # x shape: [batch_size, 15, 10, 9]
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class NNetWrapper:
    def __init__(self, game, batch_size=2):
        self.board_y, self.board_x = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.in_channels = game.getInputChannels()

        self.nnet = AlphaZeroXiangqiNet(self.in_channels, self.action_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)
        self.batch_size = batch_size

    def train(self, examples):
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=0.001)

        for epoch in range(10):
            self.nnet.train()
            for batch_idx in range(0, len(examples), self.batch_size):
                sample = examples[batch_idx:batch_idx + self.batch_size]
                boards, pis, vs = list(zip(*sample))

                boards = np.stack([self.board2cnn(board) for board in boards], axis=0)
                boards = torch.FloatTensor(boards).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).to(self.device)

                out_pi, out_v = self.nnet(boards)

                l_pi = F.cross_entropy(out_pi, target_pis)
                l_v = F.mse_loss(out_v.reshape(-1), target_vs)
                total_loss = l_pi + l_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        self.nnet.eval()
        board = self.board2cnn(board)
        board = torch.FloatTensor(board.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pi, v = self.nnet(board)
        return F.softmax(pi, dim=1).cpu().numpy()[0], v.item()

    def save_checkpoint(self, folder, filename):
        filepath = f"{folder}/{filename}"
        torch.save(self.nnet.state_dict(), filepath)

    def load_checkpoint(self, folder, filename):
        filepath = f"{folder}/{filename}"
        self.nnet.load_state_dict(torch.load(filepath, map_location=self.device))

    def board2cnn(self, board):
        # [h, w] -7 ~ 7 -> [h, w, 15]
        board = np.eye(15)[board + 7]
        return board.transpose((2, 0, 1))


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)
