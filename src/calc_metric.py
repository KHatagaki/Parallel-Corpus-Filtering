#===============
# 与えられたファイルの評価指標のスコアをファイルに出力するプログラム
#===============

import argparse
from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src-file', '-s', type=str, default=None)
    parser.add_argument('--tgt-file', '-t', type=str, required=True)
    parser.add_argument('--pred-file', '-p', type=str, required=True)
    parser.add_argument('--dest-file', '-d', type=str, required=True)
    parser.add_argument('--metric', '-m', choices=['bleu','bleu_ja','sari','sari_ja','rouge','errant'], required=True)

    args = parser.parse_args()

    return args

def calc_bleu(tgt_texts, pred_texts):
    from sacrebleu.metrics import BLEU    
    import sacrebleu
    score_list = []
    bleu = BLEU()

    # for tgt_text, pred_text in zip(tgt_texts, pred_texts):
    #     score_list.append(bleu.corpus_score([pred_text], [[tgt_text]]).score)

    for tgt_text, pred_text in zip(tgt_texts, pred_texts):
        score_list.append(sacrebleu.sentence_bleu(pred_text, [tgt_text],  smooth_method='exp').score)

    return score_list

def calc_bleu_ja(tgt_texts, pred_texts):
    from sacrebleu.metrics import BLEU
    import sacrebleu
    score_list = []
    bleu = BLEU(tokenize="ja-mecab")

    # for tgt_text, pred_text in zip(tgt_texts, pred_texts):
    #     score_list.append(bleu.corpus_score([pred_text], [[tgt_text]]).score)

    for tgt_text, pred_text in zip(tgt_texts, pred_texts):
        score_list.append(sacrebleu.sentence_bleu(pred_text, [tgt_text],  smooth_method='exp', tokenize='ja-mecab').score)

    return score_list

def calc_sari(src_texts, tgt_texts, pred_texts):
    from easse.sari import corpus_sari
    score_list = []  
    
    for src_text, tgt_text, pred_text in zip(src_texts, tgt_texts, pred_texts):
        score_list.append(corpus_sari(orig_sents=[src_text], sys_sents=[pred_text], refs_sents=[[tgt_text]]))

    return score_list

def calc_sari_ja(src_texts, tgt_texts, pred_texts):
    import MeCab
    from easse.sari import corpus_sari

    score_list = []  

    t = MeCab.Tagger("-Owakati")

    src_texts = list(map(lambda x: t.parse(x).replace("\n",""), src_texts))
    tgt_texts = list(map(lambda x: t.parse(x).replace("\n",""), tgt_texts))
    pred_texts = list(map(lambda x: t.parse(x).replace("\n",""), pred_texts))
    
    for src_text, tgt_text, pred_text in zip(src_texts, tgt_texts, pred_texts):
        score_list.append(corpus_sari(orig_sents=[src_text], sys_sents=[pred_text], refs_sents=[[tgt_text]]))

    return score_list

def calc_rouge(src_texts, tgt_texts, pred_texts):
    from sumeval.metrics.rouge import RougeCalculator
    score_list = [] 

    rouge = RougeCalculator(stopwords=True, lang="en")
    
    for tgt_text, pred_text in zip(tgt_texts, pred_texts):
        score_list.append(rouge.rouge_l(
                summary=pred_text,
                references=tgt_text))

    return score_list

def calc_errant(src_texts, tgt_texts, pred_texts):
    from contextlib import ExitStack
    from collections import Counter
    import errant

    score_list = []

    annotator = errant.load("en")
    cor_id = 0

    src2tgt_text = ""
    src2pred_text = ""

    for src_text, tgt_text, pred_text in zip(src_texts, tgt_texts, pred_texts):
        src_text = src_text.strip()
        tgt_text = tgt_text.strip()
        pred_text = pred_text.strip()
        src = annotator.parse(src_text, True)
        tgt = annotator.parse(tgt_text, True)
        pred = annotator.parse(pred_text, True)

        src2tgt_text += (" ".join(["S"]+[token.text for token in src])+"\n")
        src2pred_text += (" ".join(["S"]+[token.text for token in src])+"\n")
        if src.text.strip() == tgt_text:
            src2tgt_text += noop_edit(cor_id)+"\n"
        else:
            # Align the texts and extract and classify the edits
            edits = annotator.annotate(src, tgt, False, "rules")
            # Loop through the edits
            for edit in edits:
                # Write the edit to the output m2 file
                src2tgt_text += edit.to_m2(cor_id)+"\n"
        
        src2tgt_text += "\n"

        if src.text.strip() == pred_text:
            src2pred_text += noop_edit(cor_id)+"\n"
        else:
            # Align the texts and extract and classify the edits
            edits = annotator.annotate(src, pred, False, "rules")
            # Loop through the edits
            for edit in edits:
                # Write the edit to the output m2 file
                src2pred_text += edit.to_m2(cor_id)+"\n"
        
        src2pred_text += "\n"
    
    src2tgt = src2tgt_text.split("\n\n")
    src2pred = src2pred_text.split("\n\n")
    src2tgt = src2tgt[:len(src2tgt)-1]
    src2pred = src2pred[:len(src2pred)-1]
    sents = zip(src2pred, src2tgt)

    for sent_id, sent in enumerate(sents):
        best_dict = Counter({"tp":0, "fp":0, "fn":0})
        best_cats = {}

        # Simplify the edits into lists of lists
        hyp_edits = simplify_edits(sent[0])
        ref_edits = simplify_edits(sent[1])
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits)
        ref_dict = process_edits(ref_edits)
        # Evaluate edits and get best TP, FP, FN hyp+ref combo.
        count_dict, cat_dict = evaluate_edits(
            hyp_dict, ref_dict, best_dict, sent_id)
        # Merge these dicts with best_dict and best_cats
        best_dict += Counter(count_dict)
        best_cats = merge_dict(best_cats, cat_dict)
        # ### my edit
        score_list.append(list(computeFScore(best_dict["tp"], best_dict["fp"], best_dict["fn"], 0.5))[-1])  
        # ###
    
    
    return score_list

def noop_edit(id=0):
    return "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||"+str(id)

# Input: An m2 format sentence with edits.
# Output: A list of lists. Each edit: [start, end, cat, cor, coder]
def simplify_edits(sent):
    out_edits = []
    # Get the edit lines from an m2 block.
    edits = sent.split("\n")[1:]
    # Loop through the edits
    for edit in edits:
        # Preprocessing
        edit = edit[2:].split("|||") # Ignore "A " then split.
        span = edit[0].split()
        start = int(span[0])
        end = int(span[1])
        cat = edit[1]
        cor = edit[2]
        coder = int(edit[-1])
        out_edit = [start, end, cat, cor, coder]
        out_edits.append(out_edit)
    return out_edits

def process_edits(edits):
    coder_dict = {}
    # Add an explicit noop edit if there are no edits.
    if not edits: edits = [[-1, -1, "noop", "-NONE-", 0]]
    # Loop through the edits
    for edit in edits:
        # Name the edit elements for clarity
        start = edit[0]
        end = edit[1]
        cat = edit[2]
        cor = edit[3]
        coder = edit[4]
        # Add the coder to the coder_dict if necessary
        if coder not in coder_dict: coder_dict[coder] = {}

        # Optionally apply filters based on args
        # 1. UNK type edits are only useful for detection, not correction.
        if cat == "UNK": continue

        if (start, end, cor) in coder_dict[coder].keys():
            coder_dict[coder][(start, end, cor)].append(cat)
        else:
            coder_dict[coder][(start, end, cor)] = [cat]

    return coder_dict

def evaluate_edits(hyp_dict, ref_dict, best, sent_id):
    # Store the best sentence level scores and hyp+ref combination IDs
    # best_f is initialised as -1 cause 0 is a valid result.
    best_tp, best_fp, best_fn, best_f, best_hyp, best_ref = 0, 0, 0, -1, 0, 0
    best_cat = {}
    # Compare each hyp and ref combination
    for hyp_id in hyp_dict.keys():
        for ref_id in ref_dict.keys():
            # Get the local counts for the current combination.
            tp, fp, fn, cat_dict = compareEdits(hyp_dict[hyp_id], ref_dict[ref_id])
            # Compute the local sentence scores (for verbose output only)
            loc_p, loc_r, loc_f = computeFScore(tp, fp, fn, 0.5)
            # Compute the global sentence scores
            p, r, f = computeFScore(
                tp+best["tp"], fp+best["fp"], fn+best["fn"], 0.5)
            # Save the scores if they are better in terms of:
            # 1. Higher F-score
            # 2. Same F-score, higher TP
            # 3. Same F-score and TP, lower FP
            # 4. Same F-score, TP and FP, lower FN
            if     (f > best_f) or \
                (f == best_f and tp > best_tp) or \
                (f == best_f and tp == best_tp and fp < best_fp) or \
                (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn):
                best_tp, best_fp, best_fn = tp, fp, fn
                best_f, best_hyp, best_ref = f, hyp_id, ref_id
                best_cat = cat_dict
    # Save the best TP, FP and FNs as a dict, and return this and the best_cat dict
    best_dict = {"tp":best_tp, "fp":best_fp, "fn":best_fn}
    return best_dict, best_cat

def compareEdits(hyp_edits, ref_edits):    
    tp = 0    # True Positives
    fp = 0    # False Positives
    fn = 0    # False Negatives
    cat_dict = {} # {cat: [tp, fp, fn], ...}

    for h_edit, h_cats in hyp_edits.items():
        # noop hyp edits cannot be TP or FP
        if h_cats[0] == "noop": continue
        # TRUE POSITIVES
        if h_edit in ref_edits.keys():
            # On occasion, multiple tokens at same span.
            for h_cat in ref_edits[h_edit]: # Use ref dict for TP
                tp += 1
                # Each dict value [TP, FP, FN]
                if h_cat in cat_dict.keys():
                    cat_dict[h_cat][0] += 1
                else:
                    cat_dict[h_cat] = [1, 0, 0]
        # FALSE POSITIVES
        else:
            # On occasion, multiple tokens at same span.
            for h_cat in h_cats:
                fp += 1
                # Each dict value [TP, FP, FN]
                if h_cat in cat_dict.keys():
                    cat_dict[h_cat][1] += 1
                else:
                    cat_dict[h_cat] = [0, 1, 0]
    for r_edit, r_cats in ref_edits.items():
        # noop ref edits cannot be FN
        if r_cats[0] == "noop": continue
        # FALSE NEGATIVES
        if r_edit not in hyp_edits.keys():
            # On occasion, multiple tokens at same span.
            for r_cat in r_cats:
                fn += 1
                # Each dict value [TP, FP, FN]
                if r_cat in cat_dict.keys():
                    cat_dict[r_cat][2] += 1
                else:
                    cat_dict[r_cat] = [0, 0, 1]
    return tp, fp, fn, cat_dict

def computeFScore(tp, fp, fn, beta):
    p = float(tp)/(tp+fp) if fp else 1.0
    r = float(tp)/(tp+fn) if fn else 1.0
    f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

def merge_dict(dict1, dict2):
    for cat, stats in dict2.items():
        if cat in dict1.keys():
            dict1[cat] = [x+y for x, y in zip(dict1[cat], stats)]
        else:
            dict1[cat] = stats
    return dict1

def write_file(scores, dest_file):
    with open(dest_file, "w") as f:
        print(*scores, sep="\n", file=f)

def main(args):

    with open(f'{args.pred_file}','r') as f:
        pred_texts = f.readlines()

    with open(f'{args.tgt_file}','r') as f:
        tgt_texts = f.readlines()

    if not "bleu" in args.metric:
        if args.src_file == None:
            logger.error("You need source file")
            exit()
        else:
            with open(f'{args.src_file}','r') as f:
                src_texts = f.readlines() 
                src_texts = [text.strip() for text in src_texts]  

    # if len(pred_texts) != len(tgt_texts):
    #     print("ファイルの大きさが違います")
    #     exit()

    if args.metric == "bleu":
        scores = calc_bleu(tgt_texts, pred_texts)
    elif args.metric == "bleu_ja":
        scores = calc_bleu_ja(tgt_texts, pred_texts)
    elif args.metric == "sari":  
        scores = calc_sari(src_texts, tgt_texts, pred_texts)
    elif args.metric == "sari_ja":  
        scores = calc_sari_ja(src_texts, tgt_texts, pred_texts)
    elif args.metric == "rouge":  
        scores = calc_rouge(src_texts, tgt_texts, pred_texts)
    elif args.metric == "errant":  
        scores = calc_errant(src_texts, tgt_texts, pred_texts)

    # scores = [round(score,2) for score in scores]

    write_file(scores, args.dest_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)