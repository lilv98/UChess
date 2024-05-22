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
import math
import zstandard as zstd
from einops import rearrange, repeat
from utils import get_all_possible_moves, create_elo_dict, map_to_category, map_to_category_name, board_to_tensor, get_side_info
from utils import read_monthly_data_path, combine_players, filter_players, extract_clock_time, mirror_move, iterator


class UnseenPlayerDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, player2elo, all_moves_dict, elo_dict, cfg):
        
        self.all_moves_dict = all_moves_dict
        self.player2elo = player2elo
        self.elo_dict = elo_dict
        self.cfg = cfg
        self.data = data
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        fen, move, _, elo_oppo, player_name = self.data[idx]
    
        board = chess.Board(fen)
        board_input = board_to_tensor(board)
        move_input = self.all_moves_dict[move]
        player_elo = self.player2elo[player_name]
        elo_self = map_to_category(player_elo, self.elo_dict)
        elo_oppo = map_to_category(elo_oppo, self.elo_dict)
        legal_moves, _ = get_side_info(board, move, self.all_moves_dict)
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves


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


class MAIA2ModelUnseen(torch.nn.Module):
    
    def __init__(self, output_dim, elo_dict, cfg, player_elo):
        super(MAIA2ModelUnseen, self).__init__()
        checkpoint = torch.load(cfg.model_path_base)
        self.maia2_model = MAIA2Model(output_dim, elo_dict, cfg)
        self.maia2_model.load_state_dict(checkpoint['model_state_dict'])
        self.player_embedding = torch.nn.Embedding(1, cfg.elo_dim)
        self.player_embedding.weight.data.copy_(self.maia2_model.elo_embedding(torch.tensor([map_to_category(player_elo, elo_dict)])))
        self.cfg = cfg

    def forward(self, boards, elos_self, elos_oppo):

        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.maia2_model.cfg.input_channels, 8, 8)
        embs = self.maia2_model.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.maia2_model.to_patch_embedding(embs)
        x += self.maia2_model.pos_embedding
        x = self.maia2_model.dropout(x)
        
        elos_emb_self = self.player_embedding(torch.zeros_like(elos_self))
        
        elos_emb_oppo = self.maia2_model.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        x = self.maia2_model.transformer(x, elos_emb).mean(dim=1)
        
        x = self.maia2_model.last_ln(x)
        
        logits_maia = self.maia2_model.fc_1(x)
        
        return logits_maia


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


class IndividualModelUnseen(torch.nn.Module):
    
    def __init__(self, output_dim, elo_dict, cfg, player2elo, player_elo):
        super(IndividualModelUnseen, self).__init__()
        
        checkpoint_base = torch.load(cfg.model_path_base)
        self.prototype_model = IndividualModel(output_dim, elo_dict, cfg, checkpoint_base, player2elo)
        model_path_prototype = cfg.model_root_prototype + f'/prototype_{cfg.k_prototype}/step_{cfg.checkpoint_step}.pt'
        checkpoint = torch.load(model_path_prototype)
        self.prototype_model.load_state_dict(checkpoint)
        self.player_embedding = torch.nn.Embedding(1, cfg.elo_dim)
        self.player_embedding.weight.data.copy_(self.prototype_model.maia2_model.elo_embedding(torch.tensor([map_to_category(player_elo, elo_dict)])))
        self.cfg = cfg


    def forward(self, boards, elos_self, elos_oppo):
        
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.cfg.input_channels, 8, 8)
        embs = self.prototype_model.maia2_model.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.prototype_model.maia2_model.to_patch_embedding(embs)
        x += self.prototype_model.maia2_model.pos_embedding
        x = self.prototype_model.maia2_model.dropout(x)
        
        elos_emb_self = self.player_embedding(torch.zeros_like(elos_self))
        
        elos_emb_oppo = self.prototype_model.maia2_model.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        x = self.prototype_model.maia2_model.transformer(x, elos_emb).mean(dim=1)
        
        x = self.prototype_model.maia2_model.last_ln(x)
        
        logits_maia = self.prototype_model.maia2_model.fc_1(x)
        
        return logits_maia


class Maia4All(torch.nn.Module):
    
    def __init__(self, output_dim, elo_dict, cfg, player2elo, emb, player_elo, random_init=0):
        super(Maia4All, self).__init__()
        
        checkpoint_base = torch.load(cfg.model_path_base)
        self.prototype_model = IndividualModel(output_dim, elo_dict, cfg, checkpoint_base, player2elo)
        model_path_prototype = cfg.model_root_prototype + f'/prototype_{cfg.k_prototype}/step_{cfg.checkpoint_step}.pt'
        checkpoint = torch.load(model_path_prototype)
        self.prototype_model.load_state_dict(checkpoint)
        self.player_embedding = torch.nn.Embedding(1, cfg.elo_dim)
        # combined_emb = (emb + self.prototype_model.maia2_model.elo_embedding(torch.tensor([map_to_category(player_elo, elo_dict)]))) / 2
        if not random_init:
            self.player_embedding.weight.data.copy_(emb)
        self.cfg = cfg


    def forward(self, boards, elos_self, elos_oppo):
        
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.cfg.input_channels, 8, 8)
        embs = self.prototype_model.maia2_model.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.prototype_model.maia2_model.to_patch_embedding(embs)
        x += self.prototype_model.maia2_model.pos_embedding
        x = self.prototype_model.maia2_model.dropout(x)
        
        elos_emb_self = self.player_embedding(torch.zeros_like(elos_self))
        
        elos_emb_oppo = self.prototype_model.maia2_model.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        x = self.prototype_model.maia2_model.transformer(x, elos_emb).mean(dim=1)
        
        x = self.prototype_model.maia2_model.last_ln(x)
        
        logits_maia = self.prototype_model.maia2_model.fc_1(x)
        
        return logits_maia


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


def get_player_data(players, cfg):
    
    ret = {}
    
    for player in tqdm.tqdm(players):
        
        games_per_player = load_player_subfiles(cfg.data_root, player)
        avg_elo = get_avg_self_elo(games_per_player, player)
        
        ret[player] = {'elo': avg_elo, 
                       'train': games_per_player[:-cfg.N_games_test-cfg.N_games_val], 
                       'val': games_per_player[-cfg.N_games_test-cfg.N_games_val:-cfg.N_games_test], 
                       'test': games_per_player[-cfg.N_games_test:]}
    
    return ret


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


def my_collate_fn(data):

    board_inputs, move_inputs, elo_self_inputs, elo_oppo_inputs, legal_moves_inputs = zip(*data)
    
    board_inputs = torch.cat(board_inputs, dim=0)
    move_inputs = torch.cat(move_inputs, dim=0)
    elo_self_inputs = torch.cat(elo_self_inputs, dim=0)
    elo_oppo_inputs = torch.cat(elo_oppo_inputs, dim=0)
    legal_moves_inputs = torch.cat(legal_moves_inputs, dim=0)
    
    return board_inputs, move_inputs, elo_self_inputs, elo_oppo_inputs, legal_moves_inputs


def select_players(player2elo, elo_dict, k_unseen):
    
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
    
    prototype_player2elo_10 = {}
    prototype_player2elo_50 = {}
    prototype_player2elo_100 = {}
    prototype_player2elo_500 = {}
    for key in players_in_ranges:
        for player in players_in_ranges[key][:10]:
            prototype_player2elo_10[player] = player2elo[player]
        for player in players_in_ranges[key][:50]:
            prototype_player2elo_50[player] = player2elo[player]
        for player in players_in_ranges[key][:100]:
            prototype_player2elo_100[player] = player2elo[player]
        for player in players_in_ranges[key][:500]:
            prototype_player2elo_500[player] = player2elo[player]
    
    unseen_player2elo = {}
    for key in players_in_ranges:
        for player in players_in_ranges[key][-k_unseen:]:
            unseen_player2elo[player] = player2elo[player]
    
    return prototype_player2elo_10, prototype_player2elo_50, prototype_player2elo_100, prototype_player2elo_500, unseen_player2elo

def evaluate(model, dataloader):

    counter = 0
    correct_move = 0
    
    perplexities = []
    
    model.eval()
    with torch.no_grad():
        
        for boards, labels, elos_self, elos_oppo, legal_moves in dataloader:
            
            boards = boards.to(device)
            labels = labels.to(device)
            elos_self = elos_self.to(device)
            elos_oppo = elos_oppo.to(device)
            legal_moves = legal_moves.to(device)

            logits_maia = model(boards, elos_self, elos_oppo)
            logits_maia_legal = logits_maia * legal_moves
            preds = logits_maia_legal.argmax(dim=-1)
            correct_move += (preds == labels).sum().item()
            counter += len(labels)
            
            mask = (legal_moves == 1)
            masked_logits = logits_maia_legal.masked_fill(~mask, float('-inf'))
            log_probs = F.log_softmax(masked_logits, dim=-1)
            labels_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze()
            mean_log_likelihood = labels_log_probs.mean()
            perplexity = torch.exp(-mean_log_likelihood)
            perplexities.append(perplexity.item())
    
    accuracy = correct_move / counter
    avg_perplexity = sum(perplexities) / len(perplexities)
    
    return round(accuracy, 4), round(avg_perplexity, 4)


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

def select_unseen_players(player2elo, elo_dict, k_unseen):
    
    players_in_ranges = {}
    N_players_in_ranges = {}
    for key in elo_dict:
        players_in_ranges[key] = []
        N_players_in_ranges[key] = 0
    
    for player, elo in player2elo.items():
        key = map_to_category_name(elo)
        players_in_ranges[key].append(player)
        N_players_in_ranges[key] += 1

    unseen_player2elo = {}
    for key in players_in_ranges:
        for player in players_in_ranges[key][-k_unseen:]:
            unseen_player2elo[player] = player2elo[player]
    
    return unseen_player2elo


def split_data(player, cfg):
    
    ret_train, ret_val, ret_test = [], [], []
    data = load_obj(os.path.join(cfg.cache_root, f'positions/{player}.pkl'))
    
    if cfg.low_resource != 0:
        ret_train = data['data'][: cfg.low_resource]
        assert cfg.low_resource <= (len(data['data']) - cfg.batch_size_test * 2)
    else:
        ret_train = data['data'][: -cfg.batch_size_test * 2]
    
    ret_val = data['data'][-cfg.batch_size_test * 2: -cfg.batch_size_test]
    ret_test = data['data'][-cfg.batch_size_test:]
    
    return ret_train, ret_val, ret_test


def parse_args(args=None):

    parser = argparse.ArgumentParser()

    # Supporting Arguments
    parser.add_argument('--data_root', default='../data/games', type=str)
    parser.add_argument('--player_root', default='../data/players', type=str)
    parser.add_argument('--cache_root', default='../data/cache', type=str)
    parser.add_argument('--save_root', default='../tmp', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_steps', default=100, type=int)
    parser.add_argument('--val_interval', default=10, type=int)
    parser.add_argument('--tolerance', default=5, type=int)
    parser.add_argument('--base_model', default='maia4all', type=str)
    parser.add_argument('--model_root_prototype', default='/datadrive/josephtang/maia-individual-2/tmp', type=str)
    parser.add_argument('--checkpoint_step', default=200000, type=int)
    parser.add_argument('--model_path_base', default='/datadrive/josephtang/MAIA2/tmp/0.0001_8192_1e-05_MAIA2_Blitz/epoch_1_2021-05.pgn.pt', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu_id', default=1, type=int)
    parser.add_argument('--k_prototype', default=100, type=int)
    parser.add_argument('--k_unseen', default=10, type=int)
    parser.add_argument('--random_init', default=0, type=int)
    # --low_resource 0: use full history
    # 20000, 8000, 4000, 2000, 800 positions
    # 500, 200, 100, 50, 20 games
    parser.add_argument('--low_resource', default=20000, type=int)
    # parser.add_argument('--proxy', default=1, type=int)

    # Tunable Arguments
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--batch_size_train', default=1024 * 4, type=int)
    parser.add_argument('--batch_size_test', default=2048, type=int)
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
    seed_everything(cfg.seed)
    
    save_root = os.path.join(cfg.save_root, f'prototype_{cfg.k_prototype}')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    
    player2elo_all = load_obj(os.path.join(cfg.cache_root, 'player2elo_all.pkl'))
    
    players_all = list(player2elo_all.keys())
    
    prototype_player2elo = select_players_for_input(player2elo_all, elo_dict, cfg.k_prototype)
    prototype_players = list(prototype_player2elo.keys())
    
    unseen_player2elo = select_unseen_players(player2elo_all, elo_dict, cfg.k_unseen)
    unseen_players = list(unseen_player2elo.keys())
    unseen_player_dict = {player: i for i, player in enumerate(unseen_players)}
    
    if cfg.low_resource == 0:
        embs = load_obj(os.path.join(cfg.cache_root, f'embs_{cfg.k_prototype}_20000.pkl'))
    else:
        embs = load_obj(os.path.join(cfg.cache_root, f'embs_{cfg.k_prototype}_{cfg.low_resource}.pkl'))
    embs = embs.cpu()
    
    device = torch.device('cuda:' + str(cfg.gpu_id))

    for player in unseen_players:
        
        if cfg.base_model == 'maia2':
            model = MAIA2ModelUnseen(len(all_moves), elo_dict, cfg, unseen_player2elo[player])

        elif cfg.base_model == 'prototype':
            model = IndividualModelUnseen(len(all_moves), elo_dict, cfg, prototype_player2elo, unseen_player2elo[player])

        elif cfg.base_model == 'maia4all':
            emb = embs[unseen_player_dict[player]]
            if not cfg.random_init:
                model = Maia4All(len(all_moves), elo_dict, cfg, prototype_player2elo, emb, unseen_player2elo[player])
            else:
                model = Maia4All(len(all_moves), elo_dict, cfg, prototype_player2elo, None, unseen_player2elo[player], random_init=1)
        
        else:
            raise ValueError('base_model must be maia2 or prototype')
        
        model = model.to(device)
        
        data_train, data_val, data_test = split_data(player, cfg)
        
        print(len(data_train), len(data_val), len(data_test), flush=True)
        
        dataset_train = UnseenPlayerDataset(data_train, unseen_player2elo, all_moves_dict, elo_dict, cfg)
        dataset_val = UnseenPlayerDataset(data_val, unseen_player2elo, all_moves_dict, elo_dict, cfg)
        dataset_test = UnseenPlayerDataset(data_test, unseen_player2elo, all_moves_dict, elo_dict, cfg)

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
        
        criterion = torch.nn.CrossEntropyLoss()
        params_to_tune = list(model.player_embedding.parameters())
        optimizer = torch.optim.Adam(params_to_tune, lr=cfg.lr, weight_decay=cfg.wd)
        
        test_acc_before, test_perplexity_before = evaluate(model, dataloader_test)
        print(f'Player: {player}, Elo: {unseen_player2elo[player]}, Test Acc Before: {test_acc_before}, Test Perplexity Before: {test_perplexity_before}', flush=True)
        
        avg_loss = []
        best_test_acc = test_acc_before
        best_test_perplexity = test_perplexity_before
        tolerance = cfg.tolerance
        
        model.train()
        
        for step in range(cfg.max_steps):
            
            boards, moves, elos_self, elos_oppo, legal_moves = next(dataloader_train)
            
            boards = boards.to(device)
            moves = moves.to(device)
            elos_self = elos_self.to(device)
            elos_oppo = elos_oppo.to(device)
            legal_moves = legal_moves.to(device)
            
            logits_maia = model(boards, elos_self, elos_oppo)
            loss = criterion(logits_maia, moves)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss.append(loss.item())
            
            if (step + 1) % cfg.val_interval == 0 or (step + 1) < 10:
                # val_acc = evaluate(model, dataloader_val)
                test_acc, test_perplexity = evaluate(model, dataloader_test)
                avg_loss = sum(avg_loss) / len(avg_loss)
                model.train()
                
                if test_acc >= best_test_acc:
                    best_test_acc = test_acc
                
                if test_perplexity <= best_test_perplexity:
                    best_test_perplexity = test_perplexity
                # if val_acc >= best_val_acc:
                #     best_val_acc = val_acc
                # # if avg_loss < best_loss:
                # #     best_loss = avg_loss
                #     best_model = model.state_dict()
                #     tolerance = cfg.tolerance
                # else:
                #     tolerance -= 1
                #     if tolerance == 0:
                #         break
                print(f'Step: {step + 1}, Loss: {round(avg_loss, 4)}, Test Acc: {test_acc}, Best Test Acc: {best_test_acc}, Test Perplexity: {test_perplexity}, Best Test Perplexity: {best_test_perplexity}', flush=True)
                # print(f'Step: {step + 1}, Loss: {round(avg_loss, 4)}, Val Acc: {val_acc}, Best Loss: {best_loss}, Tolerance: {tolerance}', flush=True)
                # print(f'Step: {step + 1}, Loss: {round(avg_loss, 4)}, Val Acc: {val_acc}, Best Val Acc: {best_val_acc}, Tolerance: {tolerance}', flush=True)
                avg_loss = []
        print('\n', flush=True)
        
        # model.load_state_dict(best_model)
        # test_acc_after = evaluate(model, dataloader_test)
        # print(f'Player: {player}, Elo: {unseen_player2elo[player]}, Test Acc After: {test_acc_after}\n', flush=True)
    
