# Experiments Of UTC-IE

Other needed requirements:

```bash
pip install sparse
pip install --no-index torch-scatter -f https://pytorch-geo
metric.com/whl/torch-1.11.0+cu113.html
sentencepiece
six
unidecode
```

`transformers` can be installed as the newest version.

## Overall Results


| Task | Dataset | PLM | Result | Paper Result (F1) | Trivia
|---|---|---|---|---|---|
|NER|Conll2003|Roberta-base|93.17|93.5||
|NER|Conll2003|Roberta-base|92.44|93.0|-Plusformer|
|NER|OntoNotes05|Roberta-base||91.8||
|Nested NER|ACE04|Roberta-base||87.5||
|Nested NER|ACE05-Ent|Roberta-base||87.8||
|Nested NER|GENIA|BioBERT||80.5||

| Task | Dataset | PLM | Result | Paper Result (Ent.F1 / Rel.F1) | Trivia
|---|---|---|---|---|---|
|RE|ACE05-R|Bert-base||88.8/64.9||
|RE|ACE05-R|Albert-xxlarge||89.9/67.8||
|RE|SciERC|SciBert|69.01/38.45|69.0/38.8||
|Symmetric RE|ACE05-R+|Bert-base||90.2/67.5||
|Symmetric RE|SciERC+|SciBert|70.65/42.71|70.0/42.5||
|EE|ACE05-E|Deberta-large||73.5/56.5||
|EE|ACE05-E+|Deberta-large|71.46/55.69|73.4/57.7||
|EE|ERE-EN|Deberta-large||60.2/52.5||

| Task | Dataset | PLM | Result (Ent.F1 / Rel.F1 / Trig.F1 / Arg.F1)  | Paper Result (Ent.F1 / Rel.F1 / Trig.F1 / Arg.F1) | Trivia
|---|---|---|---|---|---|
|Joint IE|ACE05-E+|Bert-large|91.01/65.21/72.94/57.04|91.48/65.54/73.63/57.62||
|Joint IE|ERE-EN|Bert-large||87.30/56.92/57.88/50.91||


## NER

### Conll2003

Dataset: 

```bash
cd dataset
wget https://data.deepai.org/conll2003.zip
unzip conll2003.zip
```

```bash
python train_ner.py --lr 1e-5 -b 12 -n 30 -d dconll2003 --cross_depth 2 --cross_dim 32 --use_s2 1 --use_gelu 0 -a 3 --use_ln 0 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.15 --warmup 0.1 --use_size_embed 2
```

- Device: A800
- Time: About 16 min
- Results:
    ||P|R|F|
    |---|---|---|---|
    |Paper|93.4|93.6|93.5|
    |Mine|92.81|93.54|93.17|
- Raw Results:

    ```json
    {
        'f#f#dev': 96.62, 'rec#f#dev': 96.82, 'pre#f#dev': 96.41, 's_f#f#dev': 97.89,
        's_rec#f#dev': 98.1, 's_pre#f#dev': 97.69, 'f#f#test': 93.17, 'rec#f#test': 93.54,
        'pre#f#test': 92.81, 's_f#f#test': 95.87, 's_rec#f#test': 96.25, 's_pre#f#test': 95.5
    }
    ``` 

#### No Plusformer

```bash
python train_ner.py --lr 1e-5 -b 12 -n 30 -d dconll2003 --cross_depth 0 --cross_dim 32 --use_s2 1 --use_gelu 0 -a 3 --use_ln 0 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.15 --warmup 0.1 --use_size_embed 2
```

- Device: A800
- Time: About 23 min
- Results:
    ||P|R|F|
    |---|---|---|---|
    |Paper|93.0|93.0|93.0|
    |Mine|91.77|93.13|92.44|
- Raw Results:

    ```json
    {
        'f#f#dev': 96.11, 'rec#f#dev': 96.55, 'pre#f#dev': 95.68, 's_f#f#dev': 97.57,
        's_rec#f#dev': 98.01, 's_pre#f#dev': 97.13, 'f#f#test': 92.44, 'rec#f#test': 93.13,
        'pre#f#test': 91.77, 's_f#f#test': 95.38, 's_rec#f#test': 96.09, 's_pre#f#test': 94.68
    }
    ```

#### No Position Embedding

```bash
python train_ner.py --lr 1e-5 -b 12 -n 30 -d dconll2003 --cross_depth 2 --cross_dim 32 --use_s2 1 --use_gelu 0 -a 3 --use_ln 0 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.15 --warmup 0.1 --use_size_embed 2 --mode nopos
```

- Device: A800
- Time: About 11 min
- Results:
    ||P|R|F|
    |---|---|---|---|
    |Paper|-|-|93.10|
    |Mine|92.47|93.25|92.86|
- Raw Results:

    ```json
    {
        'f#f#dev': 96.42, 'rec#f#dev': 96.77, 'pre#f#dev': 96.07, 's_f#f#dev': 97.79,
        's_rec#f#dev': 98.15, 's_pre#f#dev': 97.44, 'f#f#test': 92.86, 'rec#f#test': 93.25,
        'pre#f#test': 92.47, 's_f#f#test': 95.63, 's_rec#f#test': 96.03, 's_pre#f#test': 95.22
    }
    ```

#### No CNN

```bash
python train_ner.py --lr 1e-5 -b 12 -n 30 -d dconll2003 --cross_depth 2 --cross_dim 32 --use_s2 1 --use_gelu 0 -a 3 --use_ln 0 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.15 --warmup 0.1 --use_size_embed 2 --mode nocnn
```

- Device: A800
- Time: About 15 min
- Results:
    ||P|R|F|
    |---|---|---|---|
    |Paper|||93.25|
    |Mine|92.32|93.43|92.87|
- Raw Results:

    ```json
    {
        'f#f#dev': 96.44, 'rec#f#dev': 96.82, 'pre#f#dev': 96.06, 's_f#f#dev': 97.78,
        's_rec#f#dev': 98.17, 's_pre#f#dev': 97.4, 'f#f#test': 92.87, 'rec#f#test': 93.43,
        'pre#f#test': 92.32, 's_f#f#test': 95.76, 's_rec#f#test': 96.33, 's_pre#f#test': 95.19
    } 
    ```

#### No Axis Aware

```bash
python train_ner.py --lr 1e-5 -b 12 -n 30 -d dconll2003 --cross_depth 2 --cross_dim 32 --use_s2 1 --use_gelu 0 -a 3 --use_ln 0 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.15 --warmup 0.1 --use_size_embed 2 --mode noaxis
```

- Device: A800
- Time: About 15 min
- Results:
    ||P|R|F|
    |---|---|---|---|
    |Paper|||93.23|
    |Mine|92.91|93.24|93.07|
- Raw Results:

    ```json
    {
        'f#f#dev': 96.64, 'rec#f#dev': 96.84, 'pre#f#dev': 96.45, 's_f#f#dev': 97.92,
        's_rec#f#dev': 98.12, 's_pre#f#dev': 97.72, 'f#f#test': 93.07, 'rec#f#test': 93.24,
        'pre#f#test': 92.91, 's_f#f#test': 95.86, 's_rec#f#test': 96.03, 's_pre#f#test': 95.7
    }
    ```

#### CNN-IE

```bash
python train_ner.py --lr 1e-5 -b 12 -n 30 -d dconll2003 --cross_depth 2 --cross_dim 32 --use_s2 1 --use_gelu 0 -a 3 --use_ln 0 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.15 --warmup 0.1 --use_size_embed 2 --mode cnnie
```

- Device: A800
- Time: About 6 min
- Results:
    ||P|R|F|
    |---|---|---|---|
    |Paper|-|-|93.32|
    |Mine|92.59|93.34|92.96|
- Raw Results:

    ```json
    {
        'f#f#dev': 96.63, 'rec#f#dev': 96.8, 'pre#f#dev': 96.46, 's_f#f#dev': 97.89,
        's_rec#f#dev': 98.06, 's_pre#f#dev': 97.72, 'f#f#test': 92.96, 'rec#f#test': 93.34,
        'pre#f#test': 92.59, 's_f#f#test': 95.73, 's_rec#f#test': 96.12, 's_pre#f#test': 95.35
    }
    ```

### Ontonotes

```bash
git clone https://github.com/yhcc/OntoNotes-5.0-NER.git
```

## Nested NER

### ACE04

### ACE05-Ent

### GENIA

## RE

### ACE05-R

### SciERC

```bash
cd dataset
wget http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz
tar zxvf sciERC_processed.tar.gz 
cp -r processed_data/json/ ./UniRE_SciERC
```

```bash
python train_re.py -d sciere --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 70 --lr 3e-5 --use_s2 1 --biaffine_size 200
```

- Device: A800
- Time: About 45 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Paper|68.0|70.1|69.0|43.6|34.9|38.8|
    |Mine|66.98|71.16|69.01|39.15|37.78|38.45|
- Raw Results:
    ```json
    {
        'f#f#dev': 72.74, 'rec#f#dev': 75.34, 'pre#f#dev': 70.31, 'r_f#f#dev': 41.7,
        'r_rec#f#dev': 41.98, 'r_pre#f#dev': 41.43, 'f#f#test': 69.01, 'rec#f#test': 71.16,
        'pre#f#test': 66.98, 'r_f#f#test': 38.45, 'r_rec#f#test': 37.78, 'r_pre#f#test': 39.15
    }
    ```

#### No Plusformer

```bash
python train_re.py -d sciere --cross_depth 0 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 70 --lr 3e-5 --use_s2 1 --biaffine_size 200
```

- Device: A800
- Time: About 35 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Paper|67.4|68.9|68.1|43.0|32.7|37.1|
    |Mine|64.98|70.15|67.47|37.28|36.86|37.07|
- Raw Results:
    ```json
    {
        'f#f#dev': 71.24, 'rec#f#dev': 74.23, 'pre#f#dev': 68.49, 'r_f#f#dev': 37.01,
        'r_rec#f#dev': 37.58, 'r_pre#f#dev': 36.46, 'f#f#test': 67.47, 'rec#f#test': 70.15,
        'pre#f#test': 64.98, 'r_f#f#test': 37.07, 'r_rec#f#test': 36.86, 'r_pre#f#test': 37.28
    } 
    ```

#### No Position Embedding

```bash
python train_re.py -d sciere --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 70 --lr 3e-5 --use_s2 1 --biaffine_size 200 --mode nopos
```

- Device: A800
- Time: About 11 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|66.31|69.61|67.92|40.46|36.14|38.18|
- Raw Results:
    ```json
    {
        'f#f#dev': 72.33, 'rec#f#dev': 74.6, 'pre#f#dev': 70.19, 'r_f#f#dev': 39.41,
        'r_rec#f#dev': 38.24, 'r_pre#f#dev': 40.65, 'f#f#test': 67.92, 'rec#f#test': 69.61,
        'pre#f#test': 66.31, 'r_f#f#test': 38.18, 'r_rec#f#test': 36.14, 'r_pre#f#test': 40.46
    }
    ```

#### No CNN

```bash
python train_re.py -d sciere --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 70 --lr 3e-5 --use_s2 1 --biaffine_size 200 --mode nocnn
```

- Device: A800
- Time: About 12 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|66.65|68.78|67.7|38.35|34.29|36.21|
- Raw Results:
    ```json
    {
        'f#f#dev': 71.67, 'rec#f#dev': 73.61, 'pre#f#dev': 69.82, 'r_f#f#dev': 40.32,
        'r_rec#f#dev': 38.46, 'r_pre#f#dev': 42.37, 'f#f#test': 67.7, 'rec#f#test': 68.78,
        'pre#f#test': 66.65, 'r_f#f#test': 36.21, 'r_rec#f#test': 34.29, 'r_pre#f#test': 38.35
    }
    ```

#### No Axis Awareness

```bash
python train_re.py -d sciere --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 70 --lr 3e-5 --use_s2 1 --biaffine_size 200 --mode noaxis
```

- Device: A800
- Time: About 15 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|67.05|68.96|67.99|36.68|35.63|36.15|
- Raw Results:
    ```json
    {
        'f#f#dev': 72.05, 'rec#f#dev': 74.23, 'pre#f#dev': 70.0, 'r_f#f#dev': 39.23,
        'r_rec#f#dev': 40.44, 'r_pre#f#dev': 38.1, 'f#f#test': 67.99, 'rec#f#test': 68.96,
        'pre#f#test': 67.05, 'r_f#f#test': 36.15, 'r_rec#f#test': 35.63, 'r_pre#f#test': 36.68
    }
    ```

#### CNN-IE

```bash
python train_re.py -d sciere --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 70 --lr 3e-5 --use_s2 1 --biaffine_size 200 --mode cnnie
```

- Device: A800
- Time: About 9 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |Mine|66.71|68.37|67.53|35.94|33.98|34.93|
- Raw Results:
    ```json
    {
        'f#f#dev': 71.96, 'rec#f#dev': 73.74, 'pre#f#dev': 70.27, 'r_f#f#dev': 38.57,
        'r_rec#f#dev': 40.44, 'r_pre#f#dev': 36.87, 'f#f#test': 67.53, 'rec#f#test': 68.37,
        'pre#f#test': 66.71, 'r_f#f#test': 34.93, 'r_rec#f#test': 33.98, 'r_pre#f#test': 35.94
    }
    ```


## Symmetric RE

### ACE05-R+

### SciERC+

Dataset is the same as `SciERC`.

```bash
python train_sym_re.py -d sciere_ --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200 --use_sym_rel 1
```

- Device: A800
- Time: About 54 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Paper|68.5|71.5|70.0|45.7|39.8|42.5|
    |Mine|68.06|71.34|69.66|44.17|40.7|42.37|
- Raw Results:
    ```json
    {
        'f#f#dev': 72.31, 'rec#f#dev': 74.72, 'pre#f#dev': 70.06, 'r_f#f#dev': 46.2,
        'r_rec#f#dev': 45.86, 'r_pre#f#dev': 46.54, 'f#f#test': 69.66, 'rec#f#test': 71.34,
        'pre#f#test': 68.06, 'r_f#f#test': 42.37, 'r_rec#f#test': 40.7, 'r_pre#f#test': 44.17
    }
    ```

#### No Plusformer

```bash
python train_sym_re.py -d sciere_ --cross_depth 0 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200 --use_sym_rel 1
```

- Device: A800
- Time: About 40 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Paper|68.0|69.6|68.8|43.5|36.2|39.5|
    |Mine|65.08|72.88|68.76|37.67|38.5|38.08|
- Raw Results:
    ```json
    {
        'f#f#dev': 70.86, 'rec#f#dev': 75.71, 'pre#f#dev': 66.59, 'r_f#f#dev': 40.63,
        'r_rec#f#dev': 42.54, 'r_pre#f#dev': 38.89, 'f#f#test': 68.76, 'rec#f#test': 72.88,
        'pre#f#test': 65.08, 'r_f#f#test': 38.08, 'r_rec#f#test': 38.5, 'r_pre#f#test': 37.67
    }
    ```

#### No Position Embedding

```bash
python train_sym_re.py -d sciere_ --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200 --use_sym_rel 1 --mode nopos
```

- Device: A800
- Time: About 25 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|65.68|71.34|68.39|39.29|39.82|38.77|
- Raw Results:
    ```json
    {
        'f#f#dev': 73.17, 'rec#f#dev': 76.33, 'pre#f#dev': 70.26, 'r_f#f#dev': 42.02,
        'r_rec#f#dev': 43.65, 'r_pre#f#dev': 40.51, 'f#f#test': 68.39, 'rec#f#test': 71.34,
        'pre#f#test': 65.68, 'r_f#f#test': 39.29, 'r_rec#f#test': 39.82, 'r_pre#f#test': 38.77
    }
    ```

#### No CNN

```bash
python train_sym_re.py -d sciere_ --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200 --use_sym_rel 1 --mode nocnn
```

- Device: A800
- Time: About 25 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|67.79|71.57|69.63|40.7|40.7|40.7|
- Raw Results:
    ```json
    {
        'f#f#dev': 73.19, 'rec#f#dev': 75.59, 'pre#f#dev': 70.95, 'r_f#f#dev': 44.54,
        'r_rec#f#dev': 45.86, 'r_pre#f#dev': 43.3, 'f#f#test': 69.63, 'rec#f#test': 71.57,
        'pre#f#test': 67.79, 'r_f#f#test': 40.7, 'r_rec#f#test': 40.7, 'r_pre#f#test': 40.7
    }
    ```

#### No Axis Awareness

```bash
python train_sym_re.py -d sciere_ --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200 --use_sym_rel 1 --mode noaxis
```

- Device: A800
- Time: About 25 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Paper|68.5|71.5|70.0|45.7|39.8|42.5|
    |Mine|68.06|71.34|69.66|44.17|40.7|42.37|
    |Mine|66.78|71.45|69.04|41.16|40.79|40.97|
- Raw Results:
    ```json
    {
        'f#f#dev': 72.83, 'rec#f#dev': 76.2, 'pre#f#dev': 69.75, 'r_f#f#dev': 45.76,
        'r_rec#f#dev': 47.15, 'r_pre#f#dev': 44.44, 'f#f#test': 69.04, 'rec#f#test': 71.45,
        'pre#f#test': 66.78, 'r_f#f#test': 40.97, 'r_rec#f#test': 40.79, 'r_pre#f#test': 41.16
    } 
    ```

#### CNN-IE

```bash
python train_sym_re.py -d sciere_ --cross_depth 3 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200 --use_sym_rel 1 --mode cnnie
```

- Device: A800
- Time: About 20 min
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|67.47|71.75|69.54|41.58|40.44|41|
- Raw Results:
    ```json
    {
        'f#f#dev': 72.53, 'rec#f#dev': 75.22, 'pre#f#dev': 70.03, 'r_f#f#dev': 45.6,
        'r_rec#f#dev': 46.78, 'r_pre#f#dev': 44.48, 'f#f#test': 69.54, 'rec#f#test': 71.75,
        'pre#f#test': 67.47, 'r_f#f#test': 41.0, 'r_rec#f#test': 40.44, 'r_pre#f#test': 41.58
    } 
    ```


## EE

### ACE05-E

```bash
git clone https://github.com/dwadden/dygiepp.git
```

### ACE05-E+

Download `data_ACE2005.zip` from [Google Drive](https://hkustconnect-my.sharepoint.com/personal/hzhangal_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhzhangal%5Fconnect%5Fust%5Fhk%2FDocuments%2FData%2F%5BACL%2D2021%5D%20Zero%2Dshot%20event%20Classification%2Fdata%5FACE2005%2Ezip&parent=%2Fpersonal%2Fhzhangal%5Fconnect%5Fust%5Fhk%2FDocuments%2FData%2F%5BACL%2D2021%5D%20Zero%2Dshot%20event%20Classification&ga=1) and then put it under `dataset/`

```bash
unzip data_ACE2005.zip
mv data_ACE2005/ ace05+
```

```bash
python train_ee.py --lr 1e-5 --cross_dim 150 --use_ln 1 --biaffine_size 300 --drop_s1_p 0.1 -b 32 --model_name microsoft/deberta-v3-large -a 1 -n 70 --cross_depth 3 -d ace05E+
```

instance of `AGGRESSIVEVOICEDAILY_20041208.2133-2` will be skipped since event id `AGGRESSIVEVOICEDAILY_20041208.2133-EV4-1` cause AssertionError due to inconsistency between text and event.

- Device: A800
- Time: About 4 hours
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Paper|70.8|76.1|73.4|57.8|57.6|57.7|
    |Mine|71.58|75.71|73.58|56.72|56.97|56.84|
- Raw Results:
    ```json
    {
        'f#f#dev': 76.15, 'rec#f#dev': 76.07, 'pre#f#dev': 76.23, 'r_f#f#dev': 64.12,
        'r_rec#f#dev': 60.74, 'r_pre#f#dev': 67.89, 'f#nest_f#dev': 76.15, 'rec#nest_f#dev': 76.07,
        'pre#nest_f#dev': 76.23, 'r_f#nest_f#dev': 64.12, 'r_rec#nest_f#dev': 60.74,
        'r_pre#nest_f#dev': 67.89, 'f#f#test': 74.07, 'rec#f#test': 77.12, 'pre#f#test': 71.24,
        'r_f#f#test': 57.37, 'r_rec#f#test': 57.72, 'r_pre#f#test': 57.04, 'f#nest_f#test': 74.07,
        'rec#nest_f#test': 77.12, 'pre#nest_f#test': 71.24, 'r_f#nest_f#test': 57.37,
        'r_rec#nest_f#test': 57.72, 'r_pre#nest_f#test': 57.04
    }
    {'f#f#dev': 76.15, 'rec#f#dev': 76.07,      evaluator.py:298
                    'pre#f#dev': 76.23, 'r_f#f#dev': 64.12,                     
                    'r_rec#f#dev': 60.74, 'r_pre#f#dev': 67.89,                 
                    'f#nest_f#dev': 76.15, 'rec#nest_f#dev':                    
                    76.07, 'pre#nest_f#dev': 76.23,                             
                    'r_f#nest_f#dev': 64.12,                                    
                    'r_rec#nest_f#dev': 60.74,                                  
                    'r_pre#nest_f#dev': 67.89, 'f#f#test':                      
                    74.07, 'rec#f#test': 77.12, 'pre#f#test':                   
                    71.24, 'r_f#f#test': 57.37, 'r_rec#f#test':                 
                    57.72, 'r_pre#f#test': 57.04,                               
                    'f#nest_f#test': 74.07, 'rec#nest_f#test':                  
                    77.12, 'pre#nest_f#test': 71.24,                            
                    'r_f#nest_f#test': 57.37,                                   
                    'r_rec#nest_f#test': 57.72,                                 
                    'r_pre#nest_f#test': 57.04}
    {
        'f#f#dev': 75.68, 'rec#f#dev': 74.15, 'pre#f#dev': 77.28, 'r_f#f#dev': 64.55,
        'r_rec#f#dev': 59.92, 'r_pre#f#dev': 69.97, 'f#nest_f#dev': 75.68, 'rec#nest_f#dev':  74.15,
        'pre#nest_f#dev': 77.28, 'r_f#nest_f#dev': 64.55, 'r_rec#nest_f#dev': 59.92,
        'r_pre#nest_f#dev': 69.97, 'f#f#test': 73.09, 'rec#f#test': 74.29, 'pre#f#test': 71.92,
        'r_f#f#test': 56.31, 'r_rec#f#test': 56.23, 'r_pre#f#test': 56.4, 'f#nest_f#test': 73.09,
        'rec#nest_f#test': 74.29, 'pre#nest_f#test': 71.92, 'r_f#nest_f#test': 56.31,
        'r_rec#nest_f#test': 56.23, 'r_pre#nest_f#test': 56.4
    }
    ```

#### No Plusformer

```bash
python train_ee.py --lr 1e-5 --cross_dim 150 --use_ln 1 --biaffine_size 300 --drop_s1_p 0.1 -b 32 --model_name microsoft/deberta-v3-large -a 1 -n 70 --cross_depth 0 -d ace05E+
```

- Device: A800
- Time: About 4.5 hours
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Paper|70.5|75.5|72.9|55.6|57.7|56.6|
    |Mine|71.18|75.71|73.37|52.68|56.82|54.68|
- Raw Results:
    ```json
    {'f#f#dev': 76.07, 'rec#f#dev': 76.07,      evaluator.py:298
                    'pre#f#dev': 76.07, 'r_f#f#dev': 61.79,                     
                    'r_rec#f#dev': 60.05, 'r_pre#f#dev': 63.62,                 
                    'f#nest_f#dev': 75.83, 'rec#nest_f#dev':                    
                    76.07, 'pre#nest_f#dev': 75.58,                             
                    'r_f#nest_f#dev': 61.44,                                    
                    'r_rec#nest_f#dev': 60.05,                                  
                    'r_pre#nest_f#dev': 62.89, 'f#f#test':                      
                    73.27, 'rec#f#test': 77.59, 'pre#f#test':                   
                    69.41, 'r_f#f#test': 55.54, 'r_rec#f#test':                 
                    58.75, 'r_pre#f#test': 52.66,                               
                    'f#nest_f#test': 73.11, 'rec#nest_f#test':                  
                    77.59, 'pre#nest_f#test': 69.12,                            
                    'r_f#nest_f#test': 55.06,                                   
                    'r_rec#nest_f#test': 58.9,                                  
                    'r_pre#nest_f#test': 51.69}
    {
        'f#f#dev': 75.5, 'rec#f#dev': 73.08, 'pre#f#dev': 78.08, 'r_f#f#dev': 61.99,
        'r_rec#f#dev': 58.69, 'r_pre#f#dev': 65.7, 'f#nest_f#dev': 75.41, 'rec#nest_f#dev': 73.08,
        'pre#nest_f#dev': 77.9, 'r_f#nest_f#dev': 61.73, 'r_rec#nest_f#dev': 58.69, 'r_pre#nest_f#dev': 65.1,
        'f#f#test': 73.37, 'rec#f#test': 75.71, 'pre#f#test': 71.18, 'r_f#f#test': 54.68,
        'r_rec#f#test': 56.82, 'r_pre#f#test': 52.68, 'f#nest_f#test': 73.29, 'rec#nest_f#test': 75.71,
        'pre#nest_f#test': 71.02, 'r_f#nest_f#test': 54.13, 'r_rec#nest_f#test': 56.82,                                 
        'r_pre#nest_f#test': 51.69
    }
    ```

#### No Position Embedding

```bash
python train_ee.py --lr 1e-5 --cross_dim 150 --use_ln 1 --biaffine_size 300 --drop_s1_p 0.1 -b 32 --model_name microsoft/deberta-v3-large -a 1 -n 70 --cross_depth 3 -d ace05E+ --mode nopos
```

- Device: A800
- Time: About 3 hours
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|66.4|77.83|71.66|52.4|56.68|54.77|
- Raw Results:
    ```json
    {
        'f#f#dev': 76.21, 'rec#f#dev': 79.06, 'pre#f#dev': 73.56, 'r_f#f#dev': 63.08,
        'r_rec#f#dev': 60.88, 'r_pre#f#dev': 65.44, 'f#nest_f#dev': 76.13,
        'rec#nest_f#dev': 79.06, 'pre#nest_f#dev': 73.41, 'r_f#nest_f#dev': 63.03,
        'r_rec#nest_f#dev': 60.88, 'r_pre#nest_f#dev': 65.35, 'f#f#test': 71.66,
        'rec#f#test': 77.83, 'pre#f#test': 66.4, 'r_f#f#test': 54.77,
        'r_rec#f#test': 56.68, 'r_pre#f#test': 52.98, 'f#nest_f#test': 71.66,
        'rec#nest_f#test': 77.83, 'pre#nest_f#test': 66.4, 'r_f#nest_f#test': 54.45,
        'r_rec#nest_f#test': 56.68, 'r_pre#nest_f#test': 52.4
    }
    ```

#### No CNN

```bash
python train_ee.py --lr 1e-5 --cross_dim 150 --use_ln 1 --biaffine_size 300 --drop_s1_p 0.1 -b 32 --model_name microsoft/deberta-v3-large -a 1 -n 70 --cross_depth 3 -d ace05E+ --mode nocnn
```

- Device: A800
- Time: About 3 hours
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|67.64|76.42|71.76|50.43|60.39|54.96|
- Raw Results:
    ```json
    {
        'f#f#dev': 75.67, 'rec#f#dev': 75.43, 'pre#f#dev': 75.91, 'r_f#f#dev': 63.98,
        'r_rec#f#dev': 61.83, 'r_pre#f#dev': 66.28, 'f#nest_f#dev': 75.35,
        'rec#nest_f#dev': 75.43, 'pre#nest_f#dev': 75.27, 'r_f#nest_f#dev': 63.71,
        'r_rec#nest_f#dev': 61.83, 'r_pre#nest_f#dev': 65.7, 'f#f#test': 71.76, 'rec#f#test': 76.42,
        'pre#f#test': 67.64, 'r_f#f#test': 54.96, 'r_rec#f#test': 60.39, 'r_pre#f#test': 50.43,
        'f#nest_f#test': 71.76, 'rec#nest_f#test': 76.42, 'pre#nest_f#test': 67.64,
        'r_f#nest_f#test': 54.55, 'r_rec#nest_f#test': 60.53, 'r_pre#nest_f#test': 49.64
    }
    ```

#### No Axis Awareness

```bash
python train_ee.py --lr 1e-5 --cross_dim 150 --use_ln 1 --biaffine_size 300 --drop_s1_p 0.1 -b 32 --model_name microsoft/deberta-v3-large -a 1 -n 70 --cross_depth 3 -d ace05E+ --mode noaxis
```

- Device: A800
- Time: About 3 hours
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|67.37|75.0|70.98|53.21|55.34|54.25|
- Raw Results:
    ```json
    {
        'f#f#dev': 76.67, 'rec#f#dev': 75.85, 'pre#f#dev': 77.51, 'r_f#f#dev': 62.89,
        'r_rec#f#dev': 60.05, 'r_pre#f#dev': 66.02, 'f#nest_f#dev': 76.67, 'rec#nest_f#dev': 75.85,
        'pre#nest_f#dev': 77.51, 'r_f#nest_f#dev': 62.99, 'r_rec#nest_f#dev': 60.19,
        r_pre#nest_f#dev': 66.07, 'f#f#test': 70.98, 'rec#f#test': 75.0, 'pre#f#test': 67.37,
        'r_f#f#test': 54.25, 'r_rec#f#test': 55.34, 'r_pre#f#test': 53.21, 'f#nest_f#test': 70.98,
        'rec#nest_f#test': 75.0, 'pre#nest_f#test': 67.37, 'r_f#nest_f#test': 54.25,
        'r_rec#nest_f#test': 55.34, 'r_pre#nest_f#test': 53.21
    } 
    ```

#### CNN-IE

```bash
python train_ee.py --lr 1e-5 --cross_dim 150 --use_ln 1 --biaffine_size 300 --drop_s1_p 0.1 -b 32 --model_name microsoft/deberta-v3-large -a 1 -n 70 --cross_depth 3 -d ace05E+ --mode cnnie
```

- Device: A800
- Time: About 3 hours
- Results:
    ||Ent.P|Ent.R|Ent.F|Rel.P|Rel.R|Rel.F|
    |---|---|---|---|---|---|---|
    |Mine|65.61|77.83|71.2|54.05|56.38|55.19|
- Raw Results:

    ```json
    {
        'f#f#dev': 75.87, 'rec#f#dev': 76.92, 'pre#f#dev': 74.84, 'r_f#f#dev': 65.36,
        'r_rec#f#dev': 61.42, 'r_pre#f#dev': 69.83, 'f#nest_f#dev': 75.87, 'rec#nest_f#dev': 76.92,
        'pre#nest_f#dev': 74.84, 'r_f#nest_f#dev': 65.36, 'r_rec#nest_f#dev': 61.42,
        'r_pre#nest_f#dev': 69.83, 'f#f#test': 71.2, 'rec#f#test': 77.83, 'pre#f#test': 65.61,
        'r_f#f#test': 55.19, 'r_rec#f#test': 56.38, 'r_pre#f#test': 54.05, 'f#nest_f#test': 71.12,
        'rec#nest_f#test': 77.83, 'pre#nest_f#test': 65.48, 'r_f#nest_f#test': 55.19,
        'r_rec#nest_f#test': 56.38, 'r_pre#nest_f#test': 54.05
    }
    {'f#f#dev': 76.59, 'rec#f#dev': 75.85,      evaluator.py:298
                    'pre#f#dev': 77.34, 'r_f#f#dev': 63.66,                     
                    'r_rec#f#dev': 59.78, 'r_pre#f#dev': 68.07,                 
                    'f#nest_f#dev': 76.59, 'rec#nest_f#dev':                    
                    75.85, 'pre#nest_f#dev': 77.34,                             
                    'r_f#nest_f#dev': 63.66,                                    
                    'r_rec#nest_f#dev': 59.78,                                  
                    'r_pre#nest_f#dev': 68.07, 'f#f#test':                      
                    72.85, 'rec#f#test': 76.89, 'pre#f#test':                   
                    69.21, 'r_f#f#test': 57.47, 'r_rec#f#test':                 
                    58.75, 'r_pre#f#test': 56.25,                               
                    'f#nest_f#test': 72.85, 'rec#nest_f#test':                  
                    76.89, 'pre#nest_f#test': 69.21,                            
                    'r_f#nest_f#test': 57.47,                                   
                    'r_rec#nest_f#test': 58.75,                                 
                    'r_pre#nest_f#test': 56.25}
    {'f#f#dev': 75.84, 'rec#f#dev': 74.79,      evaluator.py:298
                    'pre#f#dev': 76.92, 'r_f#f#dev': 63.79,                     
                    'r_rec#f#dev': 58.69, 'r_pre#f#dev': 69.87,                 
                    'f#nest_f#dev': 75.84, 'rec#nest_f#dev':                    
                    74.79, 'pre#nest_f#dev': 76.92,                             
                    'r_f#nest_f#dev': 63.79,                                    
                    'r_rec#nest_f#dev': 58.69,                                  
                    'r_pre#nest_f#dev': 69.87, 'f#f#test':                      
                    73.21, 'rec#f#test': 77.36, 'pre#f#test':                   
                    69.49, 'r_f#f#test': 57.0, 'r_rec#f#test':                  
                    58.01, 'r_pre#f#test': 56.02,                               
                    'f#nest_f#test': 73.21, 'rec#nest_f#test':                  
                    77.36, 'pre#nest_f#test': 69.49,                            
                    'r_f#nest_f#test': 57.0,                                    
                    'r_rec#nest_f#test': 58.01,                                 
                    'r_pre#nest_f#test': 56.02} 
    {'f#f#dev': 74.83, 'rec#f#dev': 72.44,      evaluator.py:298
                    'pre#f#dev': 77.4, 'r_f#f#dev': 63.36,                      
                    'r_rec#f#dev': 57.73, 'r_pre#f#dev': 70.22,                 
                    'f#nest_f#dev': 74.83, 'rec#nest_f#dev':                    
                    72.44, 'pre#nest_f#dev': 77.4,                              
                    'r_f#nest_f#dev': 63.36,                                    
                    'r_rec#nest_f#dev': 57.73,                                  
                    'r_pre#nest_f#dev': 70.22, 'f#f#test':                      
                    73.68, 'rec#f#test': 75.94, 'pre#f#test':                   
                    71.56, 'r_f#f#test': 58.66, 'r_rec#f#test':                 
                    58.31, 'r_pre#f#test': 59.01,                               
                    'f#nest_f#test': 73.68, 'rec#nest_f#test':                  
                    75.94, 'pre#nest_f#test': 71.56,                            
                    'r_f#nest_f#test': 58.66,                                   
                    'r_rec#nest_f#test': 58.31,                                 
                    'r_pre#nest_f#test': 59.01} 
    {'f#f#dev': 77.12, 'rec#f#dev': 76.71,      evaluator.py:298
                    'pre#f#dev': 77.54, 'r_f#f#dev': 63.68,                     
                    'r_rec#f#dev': 60.33, 'r_pre#f#dev': 67.43,                 
                    'f#nest_f#dev': 77.12, 'rec#nest_f#dev':                    
                    76.71, 'pre#nest_f#dev': 77.54,                             
                    'r_f#nest_f#dev': 63.64,                                    
                    'r_rec#nest_f#dev': 60.33,                                  
                    'r_pre#nest_f#dev': 67.33, 'f#f#test':                      
                    73.76, 'rec#f#test': 76.89, 'pre#f#test':                   
                    70.87, 'r_f#f#test': 56.46, 'r_rec#f#test':                 
                    57.42, 'r_pre#f#test': 55.52,                               
                    'f#nest_f#test': 73.76, 'rec#nest_f#test':                  
                    76.89, 'pre#nest_f#test': 70.87,                            
                    'r_f#nest_f#test': 56.46,                                   
                    'r_rec#nest_f#test': 57.42,                                 
                    'r_pre#nest_f#test': 55.52} 
    {'f#f#dev': 77.15, 'rec#f#dev': 77.56,      evaluator.py:298
                    'pre#f#dev': 76.74, 'r_f#f#dev': 63.35,                     
                    'r_rec#f#dev': 60.19, 'r_pre#f#dev': 66.87,                 
                    'f#nest_f#dev': 77.15, 'rec#nest_f#dev':                    
                    77.56, 'pre#nest_f#dev': 76.74,                             
                    'r_f#nest_f#dev': 63.35,                                    
                    'r_rec#nest_f#dev': 60.19,                                  
                    'r_pre#nest_f#dev': 66.87, 'f#f#test':                      
                    73.79, 'rec#f#test': 79.01, 'pre#f#test':                   
                    69.21, 'r_f#f#test': 56.29, 'r_rec#f#test':                 
                    59.05, 'r_pre#f#test': 53.78,                               
                    'f#nest_f#test': 73.79, 'rec#nest_f#test':                  
                    79.01, 'pre#nest_f#test': 69.21,                            
                    'r_f#nest_f#test': 56.29,                                   
                    'r_rec#nest_f#test': 59.05,                                 
                    'r_pre#nest_f#test': 53.78}
    ```

### ERE-EN

## Joint IE

### ACE05-E+

```bash
python train_ie.py -d ace05E+ --cross_dim 150 --lr 1e-5 -b 12 -n 70 --cross_depth 3 --use_ln 0 --drop_s1_p 0 --empty_rel_weight 0.1 --use_gelu 1 --biaffine_size 300
```

- Device: A800
- Time: About 7.5 hours
- Results:

    ||Ent.F|Rel.F|Trig.F|Arg.F|
    |---|---|---|---|---|
    |Paper|91.48|65.54|73.63|57.62|
    |Mine|91.01|65.21|72.94|57.42|
- Raw Results:
<!-- f->Trig e_f->Ent rel_f ->Rel er_f r_f->Arg-->
    ```json
    {
        'f#f#dev': 71.87, 'rec#f#dev': 73.72, 'pre#f#dev': 70.12, 'r_f#f#dev': 59.81,
        'r_rec#f#dev': 56.91, 'r_pre#f#dev': 63.03, 'e_f#f#dev': 90.0, 'e_rec#f#dev': 90.21,
        'e_pre#f#dev': 89.79, 'rel_f#f#dev': 69.0, 'rel_rec#f#dev': 66.35, 'rel_pre#f#dev': 71.88,         
        'er_f#f#dev': 59.15, 'er_rec#f#dev': 58.14, 'er_pre#f#dev': 60.2, 'f#f#test': 72.94,
        'rec#f#test': 80.42, 'pre#f#test': 66.73, 'r_f#f#test': 57.31, 'r_rec#f#test': 59.64,
        'r_pre#f#test': 55.14, 'e_f#f#test': 91.01, 'e_rec#f#test': 91.67, 'e_pre#f#test': 90.36,
        'rel_f#f#test': 65.21, 'rel_rec#f#test': 65.17, 'rel_pre#f#test': 65.25, 'er_f#f#test': 57.04,
        'er_rec#f#test': 61.57, 'er_pre#f#test': 53.14
    }
    ```

#### No Plusformer

```bash
python train_ie.py -d ace05E+ --cross_dim 150 --lr 1e-5 -b 12 -n 70 --cross_depth 0 --use_ln 0 --drop_s1_p 0 --empty_rel_weight 0.1 --use_gelu 1 --biaffine_size 300
```

- Device: A800
- Time: About 4 hours
- Results:

    ||Ent.F|Rel.F|Trig.F|Arg.F|
    |---|---|---|---|---|
    |Paper|90.72|62.94|72.99|55.68|
    |Mine|89.42|64.63|72.53|56.47|
- Raw Results:
<!-- f->Trig e_f->Ent rel_f ->Rel er_f r_f->Arg-->
    ```json
    {
        'f#f#dev': 72.86, 'rec#f#dev': 74.57, 'pre#f#dev': 71.22, 'r_f#f#dev': 58.92,
        'r_rec#f#dev': 56.22, 'r_pre#f#dev': 61.9, 'e_f#f#dev': 89.42,
        'e_rec#f#dev': 90.27, 'e_pre#f#dev': 88.59, 'rel_f#f#dev': 64.63,
        'rel_rec#f#dev': 65.52, 'rel_pre#f#dev': 63.77, 'er_f#f#dev': 59.46,
        'er_rec#f#dev': 58.28, 'er_pre#f#dev': 60.68, 'f#f#test': 72.53, 'rec#f#test': 80.66,
        'pre#f#test': 65.9, 'r_f#f#test': 56.47, 'r_rec#f#test': 58.9, 'r_pre#f#test': 54.23,
        'e_f#f#test': 90.36, 'e_rec#f#test': 91.61, 'e_pre#f#test': 89.14, 'rel_f#f#test': 62.97,
        'rel_rec#f#test': 65.17, 'rel_pre#f#test': 60.91, 'er_f#f#test': 55.18,
        'er_rec#f#test': 59.64, 'er_pre#f#test': 51.34
    }
    ```

#### No Position Embedding

```bash
python train_ie.py -d ace05E+ --cross_dim 150 --lr 1e-5 -b 12 -n 70 --cross_depth 3 --use_ln 0 --drop_s1_p 0 --empty_rel_weight 0.1 --use_gelu 1 --biaffine_size 300 --mode nopos
```

- Device: A800
- Time: About 4 hours
- Results:

    ||Ent.F|Rel.F|Trig.F|Arg.F|
    |---|---|---|---|---|
    |Mine|91.02|64.9|71.96|55.85|
- Raw Results:
<!-- f->Trig e_f->Ent rel_f ->Rel er_f r_f->Arg-->
    ```json
    {
        'f#f#dev': 73.63, 'rec#f#dev': 71.58, 'pre#f#dev': 75.79, 'r_f#f#dev': 60.06,
        'r_rec#f#dev': 54.72, 'r_pre#f#dev': 66.56, 'e_f#f#dev': 89.76, 'e_rec#f#dev': 89.83,
        'e_pre#f#dev': 89.7, 'rel_f#f#dev': 67.96, 'rel_rec#f#dev': 64.97, 'rel_pre#f#dev': 71.23,
        'er_f#f#dev': 60.56, 'er_rec#f#dev': 56.5, 'er_pre#f#dev': 65.24, 'f#f#test': 71.96,
        'rec#f#test': 75.94, 'pre#f#test': 68.37, 'r_f#f#test': 55.85, 'r_rec#f#test': 56.68,
        'r_pre#f#test': 55.04, 'e_f#f#test': 91.02, 'e_rec#f#test': 91.04, 'e_pre#f#test': 90.99,
        'rel_f#f#test': 64.9, 'rel_rec#f#test': 64.29, 'rel_pre#f#test': 65.52, 'er_f#f#test': 55.74,
        'er_rec#f#test': 58.31, 'er_pre#f#test': 53.4
    }
    ```

#### No CNN

```bash
python train_ie.py -d ace05E+ --cross_dim 150 --lr 1e-5 -b 12 -n 70 --cross_depth 3 --use_ln 0 --drop_s1_p 0 --empty_rel_weight 0.1 --use_gelu 1 --biaffine_size 300 --mode nocnn
```

- Device: A800
- Time: About 4 hours
- Results:

    ||Ent.F|Rel.F|Trig.F|Arg.F|
    |---|---|---|---|---|
    |Mine|91.04|63.42|71.77|56.38|
- Raw Results:
<!-- f->Trig e_f->Ent rel_f ->Rel er_f r_f->Arg-->
    ```json
    {
        'f#f#dev': 73.74, 'rec#f#dev': 75.0, 'pre#f#dev': 72.52, 'r_f#f#dev': 59.65, 'r_rec#f#dev': 56.22,
        'r_pre#f#dev': 63.52, 'e_f#f#dev': 89.48, 'e_rec#f#dev': 90.39, 'e_pre#f#dev': 88.6, 'rel_f#f#dev': 67.54,
        'rel_rec#f#dev': 67.03, 'rel_pre#f#dev': 68.06, 'er_f#f#dev': 60.39, 'er_rec#f#dev': 58.82,
        'er_pre#f#dev': 62.05, 'f#f#test': 71.77, 'rec#f#test': 80.66, 'pre#f#test': 64.65,
        'r_f#f#test': 56.38, 'r_rec#f#test': 58.31, 'r_pre#f#test': 54.58, 'e_f#f#test': 91.04,
        'e_rec#f#test': 92.13, 'e_pre#f#test': 89.98, 'rel_f#f#test': 63.42,
        'rel_rec#f#test': 66.67, 'rel_pre#f#test': 60.48, 'er_f#f#test': 56.25,
        'er_rec#f#test': 60.39, 'er_pre#f#test': 52.65
    }
    ```

#### No Axis Awareness

```bash
python train_ie.py -d ace05E+ --cross_dim 150 --lr 1e-5 -b 12 -n 70 --cross_depth 3 --use_ln 0 --drop_s1_p 0 --empty_rel_weight 0.1 --use_gelu 1 --biaffine_size 300 --mode noaxis
```

- Device: A800
- Time: About 7.5 hours
- Results:

    ||Ent.F|Rel.F|Trig.F|Arg.F|
    |---|---|---|---|---|
    |Mine|91.32|64.32|73.44|55.96|
- Raw Results:
<!-- f->Trig e_f->Ent rel_f ->Rel er_f r_f->Rel-->
    ```json
    ```

#### CNN-IE

```bash
python train_ie.py -d ace05E+ --cross_dim 150 --lr 1e-5 -b 12 -n 70 --cross_depth 3 --use_ln 0 --drop_s1_p 0 --empty_rel_weight 0.1 --use_gelu 1 --biaffine_size 300 --mode cnnie
```

- Device: A800
- Time: About 7.5 hours
- Results:

    ||Ent.F|Rel.F|Trig.F|Arg.F|
    |---|---|---|---|---|
    |Mine|90.71|64.32|72.91|54.49|
    |Mine|91.12|64.86|74.67|56.97|
- Raw Results:
<!-- f->Trig e_f->Ent rel_f ->Rel er_f r_f->Arg-->
    ```json
    {
        'f#f#dev': 73.25, 'rec#f#dev': 73.72, 'pre#f#dev': 72.78, 'r_f#f#dev': 60.21,
        'r_rec#f#dev': 55.68, 'r_pre#f#dev': 65.54, 'e_f#f#dev': 89.59, 'e_rec#f#dev': 89.42,
        'e_pre#f#dev': 89.76, 'rel_f#f#dev': 66.62, 'rel_rec#f#dev': 65.11, 'rel_pre#f#dev': 68.2,
        'er_f#f#dev': 60.62, 'er_rec#f#dev': 58.55, 'er_pre#f#dev': 62.85, 'f#f#test': 72.98,
        'rec#f#test': 79.01, 'pre#f#test': 67.81, 'r_f#f#test': 54.49, 'r_rec#f#test': 53.12,
        'r_pre#f#test': 55.94, 'e_f#f#test': 90.71, 'e_rec#f#test': 91.23, 'e_pre#f#test': 90.2,
        'rel_f#f#test': 64.32, 'rel_rec#f#test': 66.17, 'rel_pre#f#test': 62.57,
        'er_f#f#test': 54.79, 'er_rec#f#test': 56.82, 'er_pre#f#test': 52.9
    }
    ```

### ERE-EN

