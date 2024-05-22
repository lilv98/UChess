import chess.pgn
import chess
import pdb
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map
import os
from collections import defaultdict
import argparse
from utils import get_chunks, load_obj, save_obj, decompress_zst, read_monthly_data_path
from utils import readable_time, delete_file
import time


def process_chunks(pgn_path, pgn_chunks, verbose, num_workers):

    # this line is for debugging only
    # process_per_chunk((pgn_chunks[0][0], pgn_chunks[0][1], pgn_path))

    if verbose:
        results = process_map(
            process_per_chunk,
            [(start, end, pgn_path) for start, end in pgn_chunks],
            max_workers=num_workers,
            chunksize=1,
        )
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.map(
                process_per_chunk,
                [(start, end, pgn_path) for start, end in pgn_chunks],
            )

    ret = defaultdict(int)
    for d in results:
        for key, value in d.items():
            ret[key] += value
    
    return ret


def game_filter(game):

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
        return

    if "Rated" not in event:
        return

    if "Blitz" not in event:
        return

    return white, black


def process_per_chunk(args):

    start_pos, end_pos, pgn_path = args

    ret = {}

    with open(pgn_path, "r", encoding="utf-8") as pgn_file:

        pgn_file.seek(start_pos)

        while pgn_file.tell() < end_pos:

            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            filtered_game = game_filter(game)
            
            if filtered_game:

                white, black = filtered_game
                
                if white in ret:
                    ret[white] += 1
                else:
                    ret[white] = 1
                    
                if black in ret:
                    ret[black] += 1
                else:
                    ret[black] = 1

    return ret


def parse_args(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/datadrive2/lichess_data', type=str)
    parser.add_argument('--save_root', default='../data/players', type=str)
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
        player_counts = process_chunks(pgn_path, chunks, verbose=1, num_workers=cfg.num_workers)
        t_1 = time.time()
        print(f'Processing time: {readable_time(t_1 - t_0)}', flush=True)

        save_obj(player_counts, cfg.save_root + pgn_file.replace('.pgn', '_players.pkl'))
        
        # remove the decompressed PGN file to save storage
        delete_file(pgn_path)
    