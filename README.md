# Parallel-Corpus-Filtering
This repository describes unsupervised parallel corpus filtering using BART.
In this example, we use <a href="https://github.com/chaojiang06/wiki-auto">wiki-auto</a> for the training data and <a href='https://github.com/facebookresearch/asset'>ASSET</a> for the validation data.

## Installation & Requirements
<ul>
<li>Python = 3.7</li>
<li>Fairseq = 0.12.2</li>
</ul>

### Install Fairseq
```
git clone -b v0.12.2 https://github.com/facebookresearch/fairseq
cd fairseq
pip install --editable ./
cd ..
```

### Install EASSE
If you use a different evaluation tool, install the one that best suits your needs.
```
git clone https://github.com/feralvam/easse.git
cd easse
pip install -e .
cd ..
```

### Download BART
```
https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -xzvf bart.base.tar.gz
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
```

## Building a baseline model

### Tokenize by BPE
```
DATA_DIR=dataset/wiki
for SPLIT in train dev test
do
  for LANG in src trg
  do
    python -m fairseq.examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$DATA_DIR/$SPLIT.$LANG" \
    --outputs "$DATA_DIR/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
```

### Binalize the data
```
DATA_DIR=dataset/wiki
fairseq-preprocess \
  --source-lang "src" \
  --target-lang "trg" \
  --trainpref "${DATA_DIR}/train.bpe" \
  --validpref "${DATA_DIR}/dev.bpe" \
  --testpref "${DATA_DIR}/test.bpe" \
  --destdir "data-bin/" \
  --workers 60 \
  --srcdict bart.base/dict.txt \
  --tgtdict bart.base/dict.txt;
```

### Training
```
export CUDA_VISIBLE_DEVICES=0

fairseq-train data-bin \
    --arch bart_base --restore-file bart.base/model.pt \
    --task translation --no-epoch-checkpoints \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-08 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --lr 3e-5 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 500 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --seed 42 --fp16 --layernorm-embedding \
    --patience 5 --no-progress-bar \
    --max-tokens 4096 --update-freq 1 \
    --max-epoch 40 \
    --save-dir baselines \
    --skip-invalid-size-inputs-valid-test
```

## Obtain baseline model outputs for training data

### Inference on training data with a baseline model
```
export CUDA_VISIBLE_DEVICES=0

DATA_DIR=dataset/wiki

fairseq-interactive data-bin --batch-size 512 \
    --path baselines/checkpoint_best.pt --buffer-size 1024 \
    < $DATA_DIR/train.bpe.src > $DATA_DIR/train_row.pred

cat $DATA_DIR/train_row.pred | grep -P "^H-" | sort -V | cut -f 3- | sed 's/<<unk>>/<unk>/g' | sed 's/▁//g' > $DATA_DIR/train.pred
```

### Detokenize 
```
DATA_DIR=dataset/wiki

python -m src.multiprocessing_bpe_decoder \
  --encoder-json encoder.json \
  --vocab-bpe vocab.bpe \
  --inputs "$DATA_DIR/train.pred" \
  --outputs "$DATA_DIR/train.detok.pred" \
  --workers 60 \
  --keep-empty;
```

## Scoring & Filtering
```
python src/calc_metric.py -s dataset/wiki/train.src -t dataset/wiki/train.trg -p dataset/wiki/train.detok.pred -d scores/wiki_sari.out -m sari
```
After scoring, filtering is done with a threshold of 10%.
```
python src/metric_filter.py -s dataset/wiki/train.src -t dataset/wiki/train.trg -score scores/wiki_sari.out -p sari -data wiki -r 10
```
Once filtering is complete, you can see the deleted sentence pairs in the deleted_data directory and the remaining sentence pairs in the filtered_data directory.

<br>Parameters</br>
<ul>
<li>-s : Source file of training data</li>
<li>-t : Target file of training data</li>
<li>-p : Pred file of training data</li>
<li>-d : Scores are written in this file</li>
<li>-m : Metric you want to calculate (choices:'sari','sari_ja','bleu','bleu_ja','errant')</li>
</ul>

Then, build a model by fairseq on the filtered data and you are done!









