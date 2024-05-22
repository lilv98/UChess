import chess.pgn
import chess
import numpy as np
import pdb
from multiprocessing import Pool, cpu_count
import multiprocessing
from functools import partial
import torch
import tqdm
import argparse
from utils import load_obj, save_obj, seed_everything
import torch.nn as nn
import torch.nn.functional as F
import itertools
from tqdm.contrib.concurrent import process_map
import io
import os
import re
import pandas as pd
import time
import random
import zstandard as zstd
from einops import rearrange, repeat
from utils import get_all_possible_moves, create_elo_dict, map_to_category, map_to_category_name, board_to_tensor, get_side_info
from utils import read_monthly_data_path, combine_players, filter_players, extract_clock_time, mirror_move, iterator
import gc

class PrototypePlayerDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, players, player2elo, all_moves_dict, elo_dict, cfg):
        
        self.all_moves_dict = all_moves_dict
        self.players_dict_inv = {player: i for i, player in enumerate(players)}
        self.player2elo = player2elo
        self.elo_dict = elo_dict
        self.cfg = cfg
        self.data = np.array(data)
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        fen, move, _, elo_oppo, player_name = self.data[idx]
        elo_oppo = int(elo_oppo)
    
        board = chess.Board(fen)
        board_input = board_to_tensor(board)
        move_input = self.all_moves_dict[move]
        
        player_elo = self.player2elo[player_name]
        elo_self = map_to_category(player_elo, self.elo_dict)
        elo_oppo = map_to_category(elo_oppo, self.elo_dict)
        legal_moves, _ = get_side_info(board, move, self.all_moves_dict)
        player_idx = torch.tensor(self.players_dict_inv[player_name])
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves, player_idx


def process_per_game(game, player_name):

    white_elo = int(game.headers['WhiteElo'])
    black_elo = int(game.headers['BlackElo'])
    white = game.headers['White']
    black = game.headers['Black']

    ret = []
    board = game.board()
    moves = list(game.mainline_moves())
    if len(moves) < 10:
        return ret
    
    for i, node in enumerate(game.mainline()):
        move = moves[i]
        
        if i >= 10:
            
            comment = node.comment
            clock_info = extract_clock_time(comment)

            if i % 2 == 0:
                board_input = board.fen()
                move_input = move.uci()
                elo_self = white_elo
                elo_oppo = black_elo

            else:
                board_input = board.mirror().fen()
                move_input = mirror_move(move.uci())
                elo_self = black_elo
                elo_oppo = white_elo
            
            if clock_info and clock_info > 30:
                ret.append((board_input, move_input, elo_self, elo_oppo, player_name))
            else:
                break
                
        board.push(move)
        if i == 300:
            break
    
    if white == player_name:
        return ret[0::2]
    elif black == player_name:
        return ret[1::2]
    else:
        raise ValueError('Player not in game')


class BasicBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        mid_planes = planes
        
        self.conv1 = torch.nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_planes)
        self.conv2 = torch.nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out)

        return out


class ChessResNet(torch.nn.Module):
    
    def __init__(self, block, cfg):
        super(ChessResNet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(cfg.input_channels, cfg.dim_cnn, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(cfg.dim_cnn)
        self.layers = self._make_layer(block, cfg.dim_cnn, cfg.num_blocks_cnn)
        self.conv_last = torch.nn.Conv2d(cfg.dim_cnn, cfg.vit_length, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_last = torch.nn.BatchNorm2d(cfg.vit_length)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes, stride))
        
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.conv_last(out)
        out = self.bn_last(out)
        
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class EloAwareAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., elo_dim=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.elo_query = nn.Linear(elo_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, elo_emb):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        elo_effect = self.elo_query(elo_emb).view(x.size(0), self.heads, 1, -1)
        q = q + elo_effect

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., elo_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.elo_layers = nn.ModuleList([])
        for _ in range(depth):
            self.elo_layers.append(nn.ModuleList([
                EloAwareAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, elo_dim = elo_dim),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, elo_emb):
        for attn, ff in self.elo_layers:
            x = attn(x, elo_emb) + x
            x = ff(x) + x

        return self.norm(x)


class MAIA2Model(torch.nn.Module):
    
    def __init__(self, output_dim, elo_dict, cfg):
        super(MAIA2Model, self).__init__()
        
        self.cfg = cfg
        self.chess_cnn = ChessResNet(BasicBlock, cfg)
        
        heads = 16
        dim_head = 64
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8 * 8, cfg.dim_vit),
            nn.LayerNorm(cfg.dim_vit),
        )
        self.transformer = Transformer(cfg.dim_vit, cfg.num_blocks_vit, heads, dim_head, mlp_dim=cfg.dim_vit, dropout = 0.1, elo_dim = cfg.elo_dim * 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.vit_length, cfg.dim_vit))
        
        self.fc_1 = nn.Linear(cfg.dim_vit, output_dim)
        # self.fc_1_1 = nn.Linear(cfg.dim_vit, cfg.dim_vit)
        self.fc_2 = nn.Linear(cfg.dim_vit, output_dim + 6 + 6 + 1 + 64 + 64)
        # self.fc_2_1 = nn.Linear(cfg.dim_vit, cfg.dim_vit)
        self.fc_3 = nn.Linear(128, 1)
        self.fc_3_1 = nn.Linear(cfg.dim_vit, 128)
        
        self.elo_embedding = torch.nn.Embedding(len(elo_dict), cfg.elo_dim)
        
        self.dropout = nn.Dropout(p=0.1)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)


    def forward(self, boards, elos_self, elos_oppo):
        
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.cfg.input_channels, 8, 8)
        embs = self.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.to_patch_embedding(embs)
        x += self.pos_embedding
        x = self.dropout(x)
        
        elos_emb_self = self.elo_embedding(elos_self)
        elos_emb_oppo = self.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        x = self.transformer(x, elos_emb).mean(dim=1)
        
        x = self.last_ln(x)

        logits_maia = self.fc_1(x)
        logits_side_info = self.fc_2(x)
        logits_value = self.fc_3(self.dropout(torch.relu(self.fc_3_1(x)))).squeeze(dim=-1)
        
        return logits_maia, logits_side_info, logits_value


class IndividualModel(torch.nn.Module):
    
    def __init__(self, output_dim, elo_dict, cfg, checkpoint, player2elo):
        super(IndividualModel, self).__init__()
        
        self.maia2_model = MAIA2Model(output_dim, elo_dict, cfg)
        self.maia2_model.load_state_dict(checkpoint['model_state_dict'])
        self.player2elo_idx = {player: map_to_category(elo, elo_dict) for player, elo in player2elo.items()}
        player_elo_indices = torch.tensor(list(self.player2elo_idx.values()))
        self.player_embedding = torch.nn.Embedding(len(player2elo), cfg.elo_dim)
        self.player_embedding.weight.data.copy_(self.maia2_model.elo_embedding(player_elo_indices))
        self.cfg = cfg


    def forward(self, boards, player_idx, elos_oppo):
        
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.maia2_model.cfg.input_channels, 8, 8)
        embs = self.maia2_model.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.maia2_model.to_patch_embedding(embs)
        x += self.maia2_model.pos_embedding
        x = self.maia2_model.dropout(x)
        
        elos_emb_self = self.player_embedding(player_idx)
        
        elos_emb_oppo = self.maia2_model.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        x = self.maia2_model.transformer(x, elos_emb).mean(dim=1)
        
        x = self.maia2_model.last_ln(x)
        
        logits_maia = self.maia2_model.fc_1(x)
        
        return logits_maia
    

def evaluate(model, dataloader, players):

    id_to_player = {i: player for i, player in enumerate(players)}
    per_player_results = []
    counter = 0
    correct_move = 0
    macro_sum = 0
    
    model.eval()
    with torch.no_grad():
        
        for boards, labels, elos_self, elos_oppo, legal_moves, player_idx in tqdm.tqdm(dataloader):
            
            boards = boards.to(device)
            labels = labels.to(device)
            elos_self = elos_self.to(device)
            elos_oppo = elos_oppo.to(device)
            legal_moves = legal_moves.to(device)
            player_idx = player_idx.to(device)

            logits_maia = model(boards, player_idx, elos_oppo)

            logits_maia_legal = logits_maia * legal_moves
            preds = logits_maia_legal.argmax(dim=-1)
            
            
            correct_move += (preds == labels).sum().item()
            counter += len(labels)
            macro_sum += (preds == labels).sum().item() / len(labels)
        
            mask = (legal_moves == 1)
            masked_logits = logits_maia_legal.masked_fill(~mask, float('-inf'))
            log_probs = F.log_softmax(masked_logits, dim=-1)
            labels_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze()
            mean_log_likelihood = labels_log_probs.mean()
            perplexity = torch.exp(-mean_log_likelihood)
            
            per_player_results.append([id_to_player[player_idx[0].item()], round((preds == labels).sum().item() / len(labels), 4), round(perplexity.item(), 4)])
            
    macro_avg = round(macro_sum / len(dataloader), 4)
    
    return macro_avg, per_player_results


def get_avg_self_elo(games, player):
    
    total_elo = 0

    for game in games:
        if f'[White "{player}"]' in game:
            start = game.find('[WhiteElo "') + 11
            end = game.find('"]', start)
            elo = int(game[start:end])
            total_elo += elo

        elif f'[Black "{player}"]' in game:
            start = game.find('[BlackElo "') + 11
            end = game.find('"]', start)
            elo = int(game[start:end])
            total_elo += elo

    return round(total_elo / len(games))


def list_player_subfiles(directory, player):
    
    pattern = re.compile(rf"lichess_db_standard_rated_\d{{4}}-\d{{2}}_{player}\.pkl$")

    return [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]


def load_player_subfiles(data_root, player):
    
    files = list_player_subfiles(data_root, player)

    ret = []
    for file in sorted(files):
        games = load_obj(file)
        ret.extend(games)

    return ret


def process_player(args):
    player, cfg = args
    
    games_per_player = load_player_subfiles(cfg.data_root, player)
    avg_elo = get_avg_self_elo(games_per_player, player)
    
    all_positions = []
    for game in games_per_player:
        pgn = io.StringIO(game)
        game = chess.pgn.read_game(pgn)
        positions = process_per_game(game, player)
        all_positions.extend(positions)
    
    ret = {'player': player, 'elo': avg_elo, 'data': all_positions}
    save_obj(ret, os.path.join(cfg.cache_root, f'positions/{player}.pkl'))


def get_player_data(players, cfg):

    process_map(process_player, [(player, cfg) for player in players], max_workers=40, chunksize=1)



def get_player2elo_each(player, cfg):
    
    games_per_player = load_player_subfiles(cfg.data_root, player)
    avg_elo = get_avg_self_elo(games_per_player, player)
    
    return player, avg_elo


def get_player2elo(players, cfg):

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 8)
    partial_get_player2elo_each = partial(get_player2elo_each, cfg=cfg)
    results = list(tqdm.tqdm(pool.imap(partial_get_player2elo_each, players), total=len(players)))
    pool.close()
    pool.join()
    player2elo = {player: elo for player, elo in results}
    
    return player2elo


def select_players(player2elo, elo_dict):
    
    players_in_ranges = {}
    N_players_in_ranges = {}
    for key in elo_dict:
        players_in_ranges[key] = []
        N_players_in_ranges[key] = 0
    
    for player, elo in player2elo.items():
        key = map_to_category_name(elo)
        players_in_ranges[key].append(player)
        N_players_in_ranges[key] += 1
    
    print(N_players_in_ranges)

    prototype_player2elo = {}
    for key in players_in_ranges:
        for player in players_in_ranges[key][:500]:
            prototype_player2elo[player] = player2elo[player]
    
    return prototype_player2elo


def select_players_for_input(player2elo, elo_dict, k_prototype):
    
    players_in_ranges = {}
    N_players_in_ranges = {}
    for key in elo_dict:
        players_in_ranges[key] = []
        N_players_in_ranges[key] = 0
    
    for player, elo in player2elo.items():
        key = map_to_category_name(elo)
        players_in_ranges[key].append(player)
        N_players_in_ranges[key] += 1

    prototype_player2elo = {}
    for key in players_in_ranges:
        for player in players_in_ranges[key][:k_prototype]:
            prototype_player2elo[player] = player2elo[player]
    
    return prototype_player2elo


def combine_data(players, players_10, cfg):
    
    ret_train, ret_val, ret_test = [], [], []
    for player in tqdm.tqdm(players):
        data = load_obj(os.path.join(cfg.cache_root, f'positions/{player}.pkl'))
        
        if len(data['data']) > (cfg.max_positions + cfg.batch_size_test * 2):
            ret_train.extend(data['data'][: cfg.max_positions])
        else:
            ret_train.extend(data['data'][: -cfg.batch_size_test * 2])
    
    for player in tqdm.tqdm(players_10):
        data = load_obj(os.path.join(cfg.cache_root, f'positions/{player}.pkl'))
        ret_val.extend(data['data'][-cfg.batch_size_test * 2: -cfg.batch_size_test])
        ret_test.extend(data['data'][-cfg.batch_size_test:])
    
    return ret_train, ret_val, ret_test


def parse_args(args=None):

    parser = argparse.ArgumentParser()

    # Supporting Arguments
    parser.add_argument('--data_root', default='../data/games', type=str)
    parser.add_argument('--player_root', default='../data/players', type=str)
    parser.add_argument('--cache_root', default='../data/cache', type=str)
    parser.add_argument('--save_root', default='../tmp', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_steps', default=500000, type=int)
    parser.add_argument('--val_interval', default=5000, type=int)
    parser.add_argument('--tolerance', default=5, type=int)
    parser.add_argument('--model_path', default='/datadrive/josephtang/MAIA2/tmp/0.0001_8192_1e-05_MAIA2_Blitz/epoch_1_2021-05.pgn.pt', type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--resume_steps', default=190000, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--k_prototype', default=100, type=int)
    parser.add_argument('--max_positions', default=100000, type=int)


    # Tunable Arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-7, type=float)
    parser.add_argument('--batch_size_test', default=2048, type=int)
    parser.add_argument('--batch_size_train', default=1024 * 8, type=int)
    parser.add_argument('--first_n_moves', default=10, type=int)
    parser.add_argument('--last_n_moves', default=10, type=int)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=2, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=8, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    parser.add_argument('--side_info', default=True, type=bool)
    parser.add_argument('--side_info_coefficient', default=1, type=float)
    parser.add_argument('--value', default=True, type=bool)
    parser.add_argument('--value_coefficient', default=1, type=float)

    return parser.parse_args(args)


if __name__ == '__main__':
    
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    
    if cfg.resume:
        seed_everything(cfg.resume_steps)
    else:
        seed_everything(cfg.seed)
    
    save_root = os.path.join(cfg.save_root, f'prototype_{cfg.k_prototype}')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    
    player2elo_all = load_obj(os.path.join(cfg.cache_root, 'player2elo_all.pkl'))
    
    players_all = list(player2elo_all.keys())
    # get_player_data(players_all, cfg)
    
    prototype_player2elo = select_players_for_input(player2elo_all, elo_dict, cfg.k_prototype)
    prototype_players = list(prototype_player2elo.keys())
    
    prototype_player2elo_10 = select_players_for_input(player2elo_all, elo_dict, 10)
    prototype_players_10 = list(prototype_player2elo_10.keys())
    
    data_train, data_val, data_test = combine_data(prototype_players, prototype_players_10, cfg)
    
    dataset_train = PrototypePlayerDataset(data_train, prototype_players, prototype_player2elo, all_moves_dict, elo_dict, cfg)
    dataset_val = PrototypePlayerDataset(data_val, prototype_players, prototype_player2elo, all_moves_dict, elo_dict, cfg)
    dataset_test = PrototypePlayerDataset(data_test, prototype_players, prototype_player2elo, all_moves_dict, elo_dict, cfg)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=cfg.batch_size_train, 
                                                   shuffle=True, 
                                                   drop_last=False, 
                                                   num_workers=cfg.num_workers)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, 
                                                 batch_size=cfg.batch_size_test,
                                                 shuffle=False, 
                                                 drop_last=False, 
                                                 num_workers=cfg.num_workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, 
                                                  batch_size=cfg.batch_size_test,
                                                  shuffle=False, 
                                                  drop_last=False, 
                                                  num_workers=cfg.num_workers)
    
    dataloader_train = iterator(dataloader_train)
    
    # individual model based on maia2 model
    checkpoint = torch.load(cfg.model_path)
    device = torch.device('cuda:' + str(cfg.gpu_id))
    model = IndividualModel(len(all_moves), elo_dict, cfg, checkpoint, prototype_player2elo)
    if cfg.resume:
        model.load_state_dict(torch.load(save_root + f'/step_{cfg.resume_steps}.pt'))
    model = model.to(device)
    
    avg_loss = []
    best_val_acc = 0
    tolerance = cfg.tolerance
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    test_acc, test_acc_per_player = evaluate(model, dataloader_test, prototype_players)
    print(f'Before FT: Test Acc: {test_acc}, Test Acc Per Player: {test_acc_per_player}', flush=True)
    
    if cfg.resume:
        ranger = range(cfg.resume_steps, cfg.max_steps)
    else:
        ranger = range(cfg.max_steps)
    
    model.train()
    for step in ranger:

        boards, labels, elos_self, elos_oppo, _, player_idx = next(dataloader_train)

        boards = boards.to(device)
        labels = labels.to(device)
        elos_self = elos_self.to(device)
        elos_oppo = elos_oppo.to(device)
        player_idx = player_idx.to(device)
        
        logits_maia = model(boards, player_idx, elos_oppo)

        loss = criterion(logits_maia, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss.append(loss.item())
        
        if (step + 1) % 100 == 0:
            print(f'Step: {step + 1}', flush=True)
        
        if (step + 1) % cfg.val_interval == 0:
            
            val_acc, val_acc_per_player = evaluate(model, dataloader_val, prototype_players)
            avg_loss = sum(avg_loss) / len(avg_loss)
            model.train()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_loss = avg_loss
                best_model = model.state_dict()
                tolerance = cfg.tolerance
            else:
                tolerance -= 1
                if tolerance == 0:
                    break
                
            print(f'Step: {step + 1}, Loss: {round(avg_loss, 4)}, Val Acc: {val_acc}, Best Val Acc: {best_val_acc}, Tolerance: {tolerance}, Val Acc Per Player: {val_acc_per_player}', flush=True)
            avg_loss = []
            torch.save(model.state_dict(), save_root + f'/step_{step + 1}.pt')
    
    model.load_state_dict(best_model)
    test_acc, test_acc_per_player = evaluate(model, dataloader_test, prototype_players)
    print(f'After FT: Test Acc: {test_acc}, Test Acc Per Player: {test_acc_per_player}', flush=True)
    
