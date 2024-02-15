import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src-file', '-s', type=str, required=True)
    parser.add_argument('--tgt-file', '-t', type=str, required=True)
    parser.add_argument('--score-file', '-score', type=str, required=True)
    parser.add_argument('--prefix', '-p', type=str, required=True)
    parser.add_argument('--dataset', '-data', type=str, required=True)
    parser.add_argument('--range', '-r', type=int, required=True)

    args = parser.parse_args()

    return args


def sort_texts_score(src_texts, tgt_texts, scores):
    if not len(src_texts) == len(tgt_texts) == len(scores):
        print("原文ファイル・目的文ファイル・スコアファイルの長さが違います")
        exit()
    return zip(*sorted(zip(scores, src_texts, tgt_texts)))


def remove_lower_score(sorted_sources, sorted_targets, sorted_scores, range):
    length = len(sorted_scores)
    threshold_score = sorted_scores[int(length * range / 100)]

    clean_srcs = []
    clean_tgts = []
    noisy_srcs = []
    noisy_tgts = []

    for sorted_source, sorted_target, sorted_score in zip(sorted_sources, sorted_targets, sorted_scores):
        if sorted_score <= threshold_score:
            noisy_srcs.append(sorted_source)
            noisy_tgts.append(sorted_target)
        else:
            clean_srcs.append(sorted_source)
            clean_tgts.append(sorted_target)

    return clean_srcs, clean_tgts, noisy_srcs, noisy_tgts


def write_files(src_texts, tgt_texts, file_prefix):
    src_file_path = f'{file_prefix}.src'
    tgt_file_path = f'{file_prefix}.trg'

    with open(src_file_path, "w") as f:
        print(*src_texts, sep='\n', file=f)

    with open(tgt_file_path, "w") as f:
        print(*tgt_texts, sep='\n', file=f)


def main(args):

    with open(f'{args.src_file}','r') as f:
        src_texts = f.readlines()

    with open(f'{args.tgt_file}','r') as f:
        tgt_texts = f.readlines()

    with open(f'{args.score_file}','r') as f:
        scores = f.readlines()

    src_texts = [text.strip() for text in src_texts]
    tgt_texts = [text.strip() for text in tgt_texts]    
    scores = [float(score.strip()) for score in scores]

    sorted_scores, sorted_sources, sorted_targets = sort_texts_score(src_texts, tgt_texts, scores) 
   
    clean_srcs, clean_tgts, noisy_srcs, noisy_tgts = remove_lower_score(sorted_sources, sorted_targets, sorted_scores, args.range)

    os.makedirs(f'filtered_data/{args.dataset}', exist_ok=True)
    os.makedirs(f'deleted_data/{args.dataset}', exist_ok=True)

    write_files(clean_srcs, clean_tgts, f'filtered_data/{args.dataset}/{args.prefix}_{args.range}')
    write_files(noisy_srcs, noisy_tgts, f'deleted_data/{args.dataset}/{args.prefix}_{args.range}')

if __name__ == "__main__":
    args = parse_args()
    main(args)