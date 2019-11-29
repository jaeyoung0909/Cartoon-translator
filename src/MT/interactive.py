#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def main(args, src):
    ret = []
    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    tokenizer = None
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    start_id = 0
    inputs = src
    results = []
    for batch in make_batches(inputs, args, task, max_positions, encode_fn):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
        }
        translations = task.inference_step(generator, models, sample)
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            results.append((start_id + id, src_tokens_i, hypos))

    # sort output to match input order
    for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, args.remove_bpe)

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'],
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            hypo_str = decode_fn(hypo_str)
            ret.append(hypo_str)

    return ret


def translator(inputs, path='ckpt/checkpoint77.pt', vocab='wiki.ko.model', data='.'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.set_defaults(beam=5, bpe='sentencepiece', cpu=False, criterion='cross_entropy', data=data, dataset_impl=None,
     decoding_format=None, diverse_beam_groups=-1, diverse_beam_strength=0.5, empty_cache_freq=0, force_anneal=None, fp16=False, 
     fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', iter_decode_eos_penalty=0.0, 
     iter_decode_force_max_iter=False, iter_decode_max_iter=10, lazy_load=False, left_pad_source='True', left_pad_target='False', 
     lenpen=1, load_alignments=False, log_format=None, log_interval=1000, lr_scheduler='fixed', lr_shrink=0.1, match_source_len=False, 
     max_len_a=0, max_len_b=200, max_sentences=None, max_source_positions=1024, max_target_positions=1024, max_tokens=None, 
     memory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, model_overrides='{}', momentum=0.99, nbest=1, no_beamable_mm=False, 
     no_early_stop=False, no_progress_bar=False, no_repeat_ngram_size=0, num_shards=1, num_workers=1, optimizer='nag', 
     path=path, prefix_size=0, print_alignment=False, print_step=False, quiet=False, raw_text=False, remove_bpe=None, 
     replace_unk=None, required_batch_size_multiple=8, results_path=None, retain_iter_history=False, sacrebleu=False, sampling=False, 
     sampling_topk=-1, sampling_topp=-1.0, score_reference=False, seed=1, sentencepiece_vocab=vocab, shard_id=0, 
     skip_invalid_size_inputs_valid_test=False, source_lang='ko', target_lang='en', task='translation', temperature=1.0, 
     tensorboard_logdir='', threshold_loss_scale=None, tokenizer=None, truncate_source=False, unkpen=0, unnormalized=False, 
     upsample_primary=1, user_dir=None, warmup_updates=0, weight_decay=0.0)
    args = parser.parse_args([])
    return main(args, inputs)
    


if __name__ == '__main__':
    print(translator(["안녕하세요 제 이름은 조쉬구요. 영국 남자에요.", "기수형 자러 가자!", "그냥 이렇게 쓸까", "이정도면 되겠지"]))
