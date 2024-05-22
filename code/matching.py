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


class EmbMatchingDatasetTrain(torch.utils.data.Dataset):
    
    def __init__(self, data, players, cfg):
        
        self.players_dict = {i: player for i, player in enumerate(players)}
        self.cfg = cfg
        self.data = data
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        player_name = self.players_dict[idx]
        data_train = self.data[player_name]['train']
        
        start = random.randint(0, len(data_train) - self.cfg.length)
        end = start + self.cfg.length
        
        data = data_train[start: end]
        
        boards_before = []
        boards_after = []
        
        for fen, move, _, _, _ in data:
            
            board = chess.Board(fen)
            board_input_before = board_to_tensor(board)
            boards_before.append(board_input_before)
            
            board.push(chess.Move.from_uci(move))
            board_input_after = board_to_tensor(board)
            boards_after.append(board_input_after)
            
        return torch.stack(boards_before), torch.stack(boards_after), idx


class EmbMatchingDatasetVal(torch.utils.data.Dataset):
    
    def __init__(self, data, players, cfg):
        
        self.players_dict = {i: player for i, player in enumerate(players)}
        self.cfg = cfg
        self.data = data
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        player_name = self.players_dict[idx]

        data = self.data[player_name]['val']
        
        boards_before = []
        boards_after = []
        
        for fen, move, _, _, _ in data:
            
            board = chess.Board(fen)
            board_input_before = board_to_tensor(board)
            boards_before.append(board_input_before)
            
            board.push(chess.Move.from_uci(move))
            board_input_after = board_to_tensor(board)
            boards_after.append(board_input_after)
        
        return torch.stack(boards_before), torch.stack(boards_after), idx


class EmbMatchingDatasetTest(torch.utils.data.Dataset):
    
    def __init__(self, data, players, cfg):
        
        self.players_dict = {i: player for i, player in enumerate(players)}
        self.cfg = cfg
        self.data = data
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        player_name = self.players_dict[idx]

        data = self.data[player_name]['train']
        
        boards_before = []
        boards_after = []
        
        for fen, move, _, _, _ in data:
            
            board = chess.Board(fen)
            board_input_before = board_to_tensor(board)
            boards_before.append(board_input_before)
            
            board.push(chess.Move.from_uci(move))
            board_input_after = board_to_tensor(board)
            boards_after.append(board_input_after)
        
        return torch.stack(boards_before), torch.stack(boards_after), idx


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


class EmbMatchingTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.elo_layers = nn.ModuleList([])
        for _ in range(depth):
            self.elo_layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.elo_layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class EmbMatchingModel(torch.nn.Module):
    def __init__(self, cfg, all_moves, elo_dict, player2elo):
        super(EmbMatchingModel, self).__init__()

        checkpoint_base = torch.load(cfg.model_path_base)
        self.prototype_model = IndividualModel(len(all_moves), elo_dict, cfg, checkpoint_base, player2elo)
        model_path_prototype = cfg.model_root_prototype + f'/prototype_{cfg.k_prototype}/step_{cfg.checkpoint_step}.pt'
        checkpoint = torch.load(model_path_prototype)
        self.prototype_model.load_state_dict(checkpoint)
        self.player_embedding = self.prototype_model.player_embedding
        self.cnn = self.prototype_model.maia2_model.chess_cnn
        self.emb_matching_transformer = EmbMatchingTransformer(dim=512, 
                                                               depth=2, 
                                                               heads=8, 
                                                               dim_head=64, 
                                                               mlp_dim=512, 
                                                               dropout=0.1)
        self.fc_concat = nn.Linear(512*2, 512)
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.length, 512))
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.cfg = cfg
        
    
    def forward(self, boards_before, boards_after):
        
        
        boards_before_inputs = boards_before.view(boards_before.size(0) * boards_before.size(1), boards_before.size(2), boards_before.size(3), boards_before.size(4))
        boards_before_embs = self.cnn(boards_before_inputs).view(boards_before.size(0), boards_before.size(1), -1)
        
        boards_after_inputs = boards_after.view(boards_after.size(0) * boards_after.size(1), boards_after.size(2), boards_after.size(3), boards_after.size(4))
        boards_after_embs = self.cnn(boards_after_inputs).view(boards_after.size(0), boards_after.size(1), -1)
        
        board_embs = torch.cat((boards_before_embs, boards_after_embs), dim=-1)
        board_embs = torch.relu(self.fc_concat(board_embs))
        
        x = self.pos_embedding + board_embs
        x = self.emb_matching_transformer(x).mean(dim=1)
        x = self.mlp(x)
        
        return x


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


def my_collate_fn(data):

    boards_before, boards_after, idx = zip(*data)

    boards_before = torch.stack(boards_before)
    boards_after = torch.stack(boards_after)
    idx = torch.tensor(idx)
    
    return boards_before, boards_after, idx


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


def combine_data(players, cfg):
    
    ret = {}
    for player in tqdm.tqdm(players):
        
        ret[player] = {}
        data = load_obj(os.path.join(cfg.cache_root, f'positions/{player}.pkl'))
        
        if len(data['data']) > (cfg.max_positions + cfg.eval_positions * 2):
            ret[player]['train'] = data['data'][: cfg.max_positions]
        else:
            ret[player]['train'] = data['data'][: -cfg.eval_positions * 2]

        ret[player]['val'] = data['data'][-cfg.eval_positions * 2: -cfg.eval_positions]
        ret[player]['test'] = data['data'][-cfg.eval_positions:]
    
    return ret


def combine_data_test(players, cfg, low_resource):
    
    ret = {}
    for player in tqdm.tqdm(players):
        
        ret[player] = {}
        data = load_obj(os.path.join(cfg.cache_root, f'positions/{player}.pkl'))
        
        ret[player]['train'] = data['data'][: low_resource]
    
    return ret


def evaluate(model, dataloader):
    
    correct_mean_counter = 0
    correct_1_counter = 0
    
    with torch.no_grad():
        for boards_before, boards_after, idx in tqdm.tqdm(dataloader):
            
            boards_before = boards_before.to(device)
            boards_after = boards_after.to(device)
            idx = idx.to(device)
            
            N_sequence = boards_before.size(1) // cfg.length
            boards_before = boards_before[:, :N_sequence * cfg.length]
            boards_after = boards_after[:, :N_sequence * cfg.length]
            
            boards_before_splited = boards_before.view(-1, cfg.length, boards_before.size(2), boards_before.size(3), boards_before.size(4))
            boards_after_splited = boards_after.view(-1, cfg.length, boards_after.size(2), boards_after.size(3), boards_after.size(4))
            
            embs = model(boards_before_splited, boards_after_splited)
            
            embs = embs.view(len(idx), -1, embs.size(-1))
            
            embs_mean = embs.mean(dim=1)
            embs_1 = embs[:, -1, :]
            
            candidates = model.player_embedding.weight.data
            
            logits_mean = (candidates * embs_mean.unsqueeze(dim=1)).sum(dim=-1)
            logits_1 = (candidates * embs_1.unsqueeze(dim=1)).sum(dim=-1)
            
            correct_mean = (logits_mean.argmax(dim=-1) == idx).sum().item()
            correct_1 = (logits_1.argmax(dim=-1) == idx).sum().item()
            
            correct_mean_counter += correct_mean
            correct_1_counter += correct_1
        
        acc_mean = round(correct_mean_counter / len(dataloader.dataset), 4)
        acc_1 = round(correct_1_counter / len(dataloader.dataset), 4)
        
        print(f'Accuracy Mean: {acc_mean}, Accuracy 1: {acc_1}', flush=True)
    
    return acc_mean, acc_1


def inference(model, dataloader):
    
    ret = []
    model.train()
    with torch.no_grad():
        for boards_before, boards_after, idx in tqdm.tqdm(dataloader):
            
            boards_before = boards_before.to(device)
            boards_after = boards_after.to(device)
            
            N_sequence = boards_before.size(1) // cfg.length
            boards_before = boards_before[:, :N_sequence * cfg.length]
            boards_after = boards_after[:, :N_sequence * cfg.length]
            
            boards_before_splited = boards_before.view(-1, cfg.length, boards_before.size(2), boards_before.size(3), boards_before.size(4))
            boards_after_splited = boards_after.view(-1, cfg.length, boards_after.size(2), boards_after.size(3), boards_after.size(4))
            
            embs = model(boards_before_splited, boards_after_splited)
            
            embs = embs.view(len(idx), -1, embs.size(-1))
            embs_mean = embs.mean(dim=1)
            
            candidates = model.player_embedding.weight.data
            logits_mean = (candidates * embs_mean.unsqueeze(dim=1)).sum(dim=-1)
            
            topk_logits, topk_indices = logits_mean.topk(10, dim=-1)
            embs_prototype = model.player_embedding.weight.data[topk_indices]
            probs = torch.softmax(topk_logits, dim=1)
            emb = (embs_prototype * probs.unsqueeze(dim=-1)).sum(dim=1)
            
            ret.append(emb)
    
    return torch.cat(ret)


def parse_args(args=None):

    parser = argparse.ArgumentParser()

    # Supporting Arguments
    parser.add_argument('--data_root', default='../data/games', type=str)
    parser.add_argument('--player_root', default='../data/players', type=str)
    parser.add_argument('--cache_root', default='../data/cache', type=str)
    parser.add_argument('--save_root', default='../tmp', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_steps', default=50000, type=int)
    parser.add_argument('--val_interval', default=1000, type=int)
    parser.add_argument('--model_root_prototype', default='../tmp', type=str)
    parser.add_argument('--checkpoint_step', default=200000, type=int)
    parser.add_argument('--model_path_base', default='/datadrive/josephtang/MAIA2/tmp/0.0001_8192_1e-05_MAIA2_Blitz/epoch_1_2021-05.pgn.pt', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu_id', default=1, type=int)
    parser.add_argument('--k_prototype', default=100, type=int)
    parser.add_argument('--k_unseen', default=10, type=int)
    parser.add_argument('--max_positions', default=100000, type=int)
    parser.add_argument('--eval_positions', default=400 * 4, type=int)
    parser.add_argument('--length', default=400, type=int)
    parser.add_argument('--inference', default=1, type=int)
    parser.add_argument('--inference_step', default=15000, type=int)
    parser.add_argument('--low_resource', default=800, type=int)

    # Tunable Arguments
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-7, type=float)
    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_test', default=4, type=int)
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
    
    save_root = cfg.save_root + f'/matching_{cfg.k_prototype}'
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
    
    unseen_player2elo = select_unseen_players(player2elo_all, elo_dict, cfg.k_unseen)
    unseen_players = list(unseen_player2elo.keys())
    
    device = torch.device(f'cuda:{cfg.gpu_id}')
    
    if not cfg.inference:
        data = combine_data(prototype_players, cfg)
        dataset_train = EmbMatchingDatasetTrain(data, prototype_players, cfg)
        dataset_val = EmbMatchingDatasetVal(data, prototype_players, cfg)
        
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=cfg.batch_size_train,
                                                        shuffle=True,
                                                        drop_last=False,
                                                        num_workers=cfg.num_workers,
                                                        collate_fn=my_collate_fn)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                        batch_size=cfg.batch_size_test,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        num_workers=cfg.num_workers,
                                                        collate_fn=my_collate_fn)
        dataloader_train = iterator(dataloader_train)
    
    model = EmbMatchingModel(cfg, all_moves, elo_dict, prototype_player2elo)
    if cfg.inference:
        model.load_state_dict(torch.load(os.path.join(save_root, f'step_{cfg.inference_step}.pt')))
    model = model.to(device)

    if cfg.inference:
        evaluate(model, dataloader_val)
        for low_resource in [800, 2000, 8000, 20000]:
            test_data = combine_data_test(unseen_players, cfg, low_resource)
            dataset_test = EmbMatchingDatasetTest(test_data, unseen_players, cfg)
            dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                            batch_size=1,
                                                            shuffle=False,
                                                            drop_last=False,
                                                            num_workers=cfg.num_workers,
                                                            collate_fn=my_collate_fn)
            embs = inference(model, dataloader_test)
            save_obj(embs, os.path.join(cfg.cache_root, f'embs_{cfg.k_prototype}_{low_resource}.pkl'))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    
    model.train()
    avg_loss = []
    best_acc = 0
    
    for step in range(cfg.max_steps):
        boards_before, boards_after, idx = next(dataloader_train)

        boards_before = boards_before.to(device)
        boards_after = boards_after.to(device)
        idx = idx.to(device)
        embs = model(boards_before, boards_after)
        
        candidates = model.player_embedding.weight.data
        logits = (candidates * embs.unsqueeze(dim=1)).sum(dim=-1)
        
        loss = criterion(logits, idx)
        avg_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            print(f'Step: {step + 1}, Loss: {round(sum(avg_loss) / len(avg_loss), 4)}', flush=True)
            avg_loss = []
            
        if (step + 1) % cfg.val_interval == 0:
            acc_mean, acc_1 = evaluate(model, dataloader_val)
            model.train()
            if acc_mean > best_acc:
                best_acc = acc_mean
            print(f'Step: {step + 1}, Acc Mean: {acc_mean}, Acc 1: {acc_1}, Best Acc Mean: {best_acc}', flush=True)
            torch.save(model.state_dict(), os.path.join(save_root, f'step_{step + 1}.pt'))
            
            
