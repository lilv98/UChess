import chess.pgn
import chess
import pdb
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map
import pandas as pd
import sys
from collections import defaultdict
import tqdm
import re
import os
import argparse
from utils import decompress_zst, get_chunks, load_obj, save_obj, read_monthly_data_path
from utils import readable_time, delete_file, filter_players, combine_players
from collections import defaultdict
import time


def process_chunks(pgn_path, pgn_chunks, players, verbose, num_workers):

    # this line is for debugging only
    # ret = process_per_chunk((pgn_chunks[0][0], pgn_chunks[0][1], pgn_path, players))

    if verbose:
        results = process_map(
            process_per_chunk,
            [(start, end, pgn_path, players) for start, end in pgn_chunks],
            max_workers=num_workers,
            chunksize=1,
        )
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.map(
                process_per_chunk,
                [(start, end, pgn_path, players) for start, end in pgn_chunks],
            )

    result_dict = defaultdict(list)
    for d in tqdm.tqdm(results):
        for key, value in d.items():
            result_dict[key].extend(value)
    result_dict = dict(result_dict)
    
    return result_dict


def game_filter(game, players):

    ret = [0, 0]
    
    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")
    white_elo = game.headers.get("WhiteElo", "?")
    black_elo = game.headers.get("BlackElo", "?")
    event = game.headers.get("Event", "?")

    if (
        white == "?"
        or black == "?"
        or white_elo == "?"
        or black_elo == "?"
        or event == "?"
    ):
        return ret

    if "Rated" not in event:
        return ret

    if "Blitz" not in event:
        return ret
    
    if white in players:
        ret[0] = white
    
    if black in players:
        ret[1] = black

    return ret


def process_per_chunk(args):

    start_pos, end_pos, pgn_path, players = args

    ret = {}

    with open(pgn_path, "r", encoding="utf-8") as pgn_file:

        pgn_file.seek(start_pos)
        
        while pgn_file.tell() < end_pos:

            game = chess.pgn.read_game(pgn_file)
            
            if game is None:
                break
            
            white, black = game_filter(game, players)
            
            if white:
                if white in ret:
                    ret[white].append(str(game))
                else:
                    ret[white] = [str(game)]
            
            if black:
                if black in ret:
                    ret[black].append(str(game))
                else:
                    ret[black] = [str(game)]

    return ret


def parse_args(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/datadrive2/lichess_data', type=str)
    parser.add_argument('--player_root', default='../data/players', type=str)
    parser.add_argument('--save_root', default='../data/games', type=str)
    parser.add_argument('--start_year', default=2023, type=int)
    parser.add_argument('--start_month', default=1, type=int)
    parser.add_argument('--end_year', default=2023, type=int)
    parser.add_argument('--end_month', default=12, type=int)
    parser.add_argument('--num_workers', default=40, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--chunk_size', default=20000, type=int)

    return parser.parse_args(args)


if __name__ == '__main__':
    
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    pgn_files = read_monthly_data_path(cfg.start_year, cfg.start_month, cfg.end_year, cfg.end_month)
    
    all_players = combine_players(pgn_files, cfg)
    players = filter_players(all_players, higher=0, new=50000)
    players = set(players)
    
    # this line is for debugging only
    # pgn_files = ['/lichess_db_standard_rated_2018-05.pgn']
    
    for pgn_file in pgn_files:
        
        print(f'Processing {pgn_file}...', flush=True)
        
        pgn_path = cfg.data_root + pgn_file
    
        # check if PGN file is already decompressed
        t_0 = time.time()
        if not os.path.exists(pgn_path):
            print('Decompressing PGN file...', flush=True)
            decompress_zst(pgn_path + '.zst', pgn_path)
        else:
            print('PGN file already decompressed.', flush=True)
        t_1 = time.time()
        print(f'Decompression time: {readable_time(t_1 - t_0)}', flush=True)
        
        # check if chunks are already saved
        t_0 = time.time()
        chunks_path = pgn_path.replace('.pgn', '_chunks.pkl')
        if not os.path.exists(chunks_path):
            print('Getting chunks...', flush=True)
            chunks = get_chunks(pgn_path, chunk_size=cfg.chunk_size)
            save_obj(chunks, chunks_path)
        else:
            chunks = load_obj(chunks_path)
            print('Chunks already saved.', flush=True)
        t_1 = time.time()
        print(f'Chunking time: {readable_time(t_1 - t_0)}', flush=True)
        
        t_0 = time.time()
        games = process_chunks(pgn_path, chunks, players, verbose=cfg.verbose, num_workers=cfg.num_workers)
        t_1 = time.time()
        print(f'Processing time: {readable_time(t_1 - t_0)}', flush=True)
        
        for player in tqdm.tqdm(games):
            games_per_player = games[player]
            file_name = cfg.save_root + pgn_file.replace('.pgn', f'_{player}.pkl')
            save_obj(games_per_player, file_name)
        
        delete_file(pgn_path)
    

    