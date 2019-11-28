requirement: fairseq, sentencepiece

$ fairseq-interactive --path checkpoint77.pt . --beam 5 --source-lang ko --target-lang en --bpe sentencepiece --sentencepiece-vocab wiki.ko.model


