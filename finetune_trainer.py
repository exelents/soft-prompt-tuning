#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    AutoConfig,
    T5Tokenizer,
    HfArgumentParser,
    MBartTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.training_args import ParallelMode
from utils1 import (
    SiameseDataCollator,
    SiameseDataset,
    assert_all_frozen,
    build_compute_metrics_fn,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
)

from utils_lm import LineByLineWebNLGTextDataset
from utils_data import DataCollatorForData2TextLanguageModeling

from gptj_utils import GPTJ_PrefixTune

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to soft pretrained prefix model"}
    )
    base_model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})

    n_prompt_tokens: Optional[int] = field(
        default=5,
        metadata={
            "help": ""
        },
    )
    n_prompts: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of prompts in model to train"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    check_output_dir(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.model_name_or_path:
        gptj_model_nods = GPTJ_PrefixTune.from_pretrained(
            model_args.model_name_or_path,
            main_checkpoint_override=model_args.base_model_path
        )
    else:
        gptj_model_nods = GPTJ_PrefixTune(
            model_args.base_model_path,
            soft_emb_n_tokens=model_args.n_prompt_tokens,
            soft_emb_n_prompts=model_args.n_prompts,
        )

    config = gptj_model_nods.config

    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    # )
    # model = AutoModelForSiameseLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=".ckpt" in model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )
    tokenizer = gptj_model_nods.tokenizer
    model = gptj_model_nods
    # use task specific params
    use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        assert (
            data_args.tgt_lang is not None and data_args.src_lang is not None
        ), "mBart requires --tgt_lang and --src_lang"
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    #model.config.n_prompt_tokens
    dataset_class = LineByLineWebNLGTextDataset
    ID_PREFIX_TOKEN = list(range(model_args.n_prompts))
    print(f"ID_PREFIX_TOKEN = {ID_PREFIX_TOKEN}")

    data_collator = DataCollatorForData2TextLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
        format_mode=data_args.format_mode
    )

    # Get datasets
    train_dataset = (
        dataset_class(
            tokenizer=tokenizer,
            file_path=os.path.join(data_args.data_dir, 'train.json'),
            block_size=data_args.max_source_length,
            bos_tok=tokenizer.bos_token,
            eos_tok=tokenizer.eos_token,
            id_prefix_token=ID_PREFIX_TOKEN,
            n_prefix_tokens=model_args.n_prompt_tokens,
            )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        dataset_class(
            tokenizer=tokenizer,
            file_path=os.path.join(data_args.data_dir, 'dev.json'),
            block_size=data_args.max_source_length,
            bos_tok=tokenizer.bos_token,
            eos_tok=tokenizer.eos_token,
            id_prefix_token=ID_PREFIX_TOKEN,
            n_prefix_tokens=model_args.n_prompt_tokens,
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        dataset_class(
            tokenizer=tokenizer,
            file_path=os.path.join(data_args.data_dir, 'val.json'),
            block_size=data_args.max_source_length,
            bos_tok=tokenizer.bos_token,
            eos_tok=tokenizer.eos_token,
            id_prefix_token=ID_PREFIX_TOKEN,
            n_prefix_tokens=model_args.n_prompt_tokens,
        )
        if training_args.do_predict
        else None
    )
    # train_batch_size = training_args.per_device_eval_batch_size
    #
    # train_dataset = torch.load("/export/data/ipynb/0_replace_by_-100/train_data_qa.pt")
    # eval_dataset = toch_size = training_args.per_device_eval_batch_size
    #
    # train_dataset = torch.load("/export/data/ipynb/0_replace_by_-100/train_data_qa.pt")
    # eval_dataset = torch.load("/export/data/ipynb/0_replace_by_-100/dev_data_qa.pt")

    # quit(type(train_dataset))
    # train_dataset = DataLoader(train_dataset,
    #                            batch_size=train_batch_size,
    #                            )
    # eval_dataset = DataLoader(eval_dataset,
    #                           batch_size=train_batch_size,
    #                           )
    # train_dataset = iter(train_dataset)
    # eval_dataset = iter(eval_dataset)

    # Initialize our Trainer
    compute_metrics_fn = (
        None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )

    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        train_result = trainer.train(
            model_path=None  # model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="val", max_length=data_args.val_max_target_length, num_beams=data_args.eval_beams
        )
        metrics["val_n_objs"] = data_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():

            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_output = trainer.predict(
            test_dataset=test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.eval_beams,
        )
        metrics = test_output.metrics
        metrics["test_n_objs"] = data_args.n_test

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = lmap(str.strip, test_preds)
                write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    torch.set_num_threads(48)
    main()
