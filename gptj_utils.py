import os
import torch
import torch.nn as nn
import json
from transformers import GPTJForCausalLM, AutoConfig, AutoTokenizer
import deepspeed
import torch
import argparse
from utils import get_argument_parser
from transformers import GPTJPreTrainedModel
from typing import Union, Iterable, Tuple
from soft_embedding import SoftEmbedding
from utils1 import freeze_params


NoneType = type(None)

dist_env_1_gpu = dict(MASTER_ADDR="localhost", MASTER_PORT="10999", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
for k,v in dist_env_1_gpu.items():
    os.environ[k] = v


def get_model_config_tokenizer(model_path):
    # GPT-J 6B config
    config = AutoConfig.from_pretrained("EleutherAI/gpt-J-6B")
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.num_layers = 28
    config.num_heads = 16
    config.hidden_size = 256 * config.num_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True

    try:
        from collections.abc import MutableMapping
    except ImportError:
        from collections import MutableMapping
    from pathlib import Path

    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", config=config).to('cpu')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", add_prefix_space=True)

    return model, config, tokenizer


def get_deepspeed_engine_optimizer(model, config_filename="ds_config_stage2_gptj.json"):
    deepspeed.init_distributed(dist_backend='nccl')

    parser = get_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    args_unparsed = f"--train_batch_size 16 --deepspeed --deepspeed_config {config_filename} --output_dir ./output_dir".split()
    args = parser.parse_args(args_unparsed)
    args.local_rank = int(os.environ['LOCAL_RANK']) if args.local_rank != -1 else args.local_rank

    config_params = json.load(open(args.deepspeed_config))
    config_params['train_batch_size'] = args.train_batch_size

    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters(),
                                                         config_params=config_params)
    return model_engine, optimizer


class GPTJ_PrefixTune(GPTJPreTrainedModel):
    def __init__(self, model_path="j6b_ckpt",
                 seq_len=512,
                 soft_emb_n_tokens=5,
                 soft_emb_n_prompts=5,
                 initialize_from_vocab=True,
                 deepspeed_config=None):
        optimizer = None
        model_unwrapped, config, tokenizer = get_model_config_tokenizer(model_path)
        config.soft_emb_n_tokens = soft_emb_n_tokens
        config.soft_emb_n_prompts = soft_emb_n_prompts

        s_wte = SoftEmbedding(model_unwrapped.get_input_embeddings(),
                              n_tokens=soft_emb_n_tokens,
                              n_prompts=soft_emb_n_prompts,
                              initialize_from_vocab=initialize_from_vocab)
        model_unwrapped.set_input_embeddings(s_wte)
        freeze_params(model_unwrapped, exclude=["transformer.wte.learned_embedding"])
        if not deepspeed_config:
            model = model_unwrapped
        else:
            model, optimizer = get_deepspeed_engine_optimizer(model_unwrapped, config_filename=deepspeed_config)

        super().__init__(config)

        self.s_wte = s_wte

        self.model_unwrapped = model_unwrapped
        self.config = config
        self.model = model
        self.optimizer = optimizer

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        print(f"self.eos_token_id={self.eos_token_id}")
        print(f"self.bos_token_id={self.bos_token_id}")
        print(f"self.pad_token_id={self.pad_token_id}")

        # self.eos_token_id = tokenizer("<|endoftext|>")['input_ids'][0]
        self.seq_len = seq_len
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.deepspeed_config = deepspeed_config

        self.model_parallel = True

    def get_soft_emb_state_dict(self, state_dict=None):
        if state_dict is None:
            state_dict = self.state_dict()
        for key, val in state_dict.items():
            if 's_wte' in key and 's_wte.wte' not in key:
                print(key)
                yield key, val

    def save_pretrained(self, model_path, state_dict=None):
        print("Save soft prompt tuning model...")
        print("================================")
        os.makedirs(model_path, exist_ok=True)
        dict_to_save = dict(list(self.get_soft_emb_state_dict(state_dict=state_dict)))
        print("================================")
        print(f"saved = {dict_to_save}")
        torch.save(
            dict_to_save,
            os.path.join(model_path, 'pytorch_weights.bin')
        )
        if len(self.config.attention_types) == 2 and \
                isinstance(self.config.attention_types[1], (int, float)):
            self.config.attention_types = [self.config.attention_types, ]
        self.config.save_pretrained(model_path)

    @classmethod
    def from_pretrained(cls, model_path, main_checkpoint_override=None, **model_args):
        print("Load soft prompt tuning model...")
        config = AutoConfig.from_pretrained(model_path)

        state_dict_load = torch.load(
            os.path.join(model_path, 'pytorch_weights.bin'),
            map_location='cpu'
        )
        state_dict = {}
        for k, v in state_dict_load.items():
            if 's_wte' in k and 's_wte.wte' not in k:
                v_save = v.clone().detach()
                state_dict[k] = v_save
            del v

        if 'soft_emb_n_tokens' in model_args:
            del model_args['soft_emb_n_tokens']
        if main_checkpoint_override:
            model_args['model_path'] = main_checkpoint_override
        model = cls(
            soft_emb_n_tokens=config.soft_emb_n_tokens,
            soft_emb_n_prompts=config.soft_emb_n_prompts,
            **model_args
        )

        model.load_state_dict(
            state_dict,
            strict=False
        )
        return model

    def __call__(self, *args, **kwargs):
        _outputs = self.model(**kwargs)
        return _outputs

    def generate(
            self,
            prompt_emb_id: int,
            text: Union[str, NoneType] = None,
            input_ids: Union[torch.LongTensor, NoneType] = None,
            return_only_generated: bool = False,
            max_length: Union[int, None] = None,
            min_length: Union[int, NoneType] = None,
            do_sample: Union[bool, NoneType] = None,
            early_stopping: Union[bool, NoneType] = None,
            num_beams: Union[int, NoneType] = None,
            temperature: Union[float, NoneType] = None,
            top_k: Union[int, NoneType] = None,
            top_p: Union[float, NoneType] = None,
            repetition_penalty: Union[float, NoneType] = None,
            bad_words_ids: Union[Iterable[int], NoneType] = None,
            bos_token_id: Union[int, NoneType] = None,
            pad_token_id: Union[int, NoneType] = None,
            eos_token_id: Union[int, NoneType] = None,
            length_penalty: Union[float, NoneType] = None,
            no_repeat_ngram_size: Union[int, NoneType] = None,
            num_return_sequences: Union[int, NoneType] = None,
            decoder_start_token_id: Union[int, NoneType] = None,
            use_cache: Union[bool, NoneType] = None,
            return_tensor=False,
            **model_kwargs):
        if text is not None:
            input_ids = torch.cuda.LongTensor([self.tokenizer(text)['input_ids']])

        prompt_ids = torch.tensor(
            [[prompt_emb_id] * self.s_wte.n_tokens] * input_ids.shape[0],
            dtype=torch.long
        ).to(input_ids.device)

        input_ids = torch.cat([prompt_ids, input_ids], 1).to(self.s_wte.wte.weight.device)

        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if self.deepspeed_config is None:
            input_ids.to('cuda')
        res = super().generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            **model_kwargs
        )
        if self.deepspeed_config is None or return_tensor:
            res.detach().to('cpu')
        if return_only_generated:
            res = res[..., input_ids.shape[1]:]
        if return_tensor:
            return res
        return list(map(self.tokenizer.decode, res.tolist()))


if __name__ == "__main__":
    # test
    gptj = GPTJ_PrefixTune(
        '/export/data/gptj/j6b_ckpt/',
        deepspeed_config='ds_config_stage2_gptj_gen.json',
    )
    gptj.save_pretrained('./save_test')
    del gptj

    gptj = GPTJ_PrefixTune.from_pretrained(
        './save_test',
        main_checkpoint_override="/export/data/gptj/j6b_ckpt/",
        deepspeed_config='ds_config_stage2_gptj_gen.json',
    )
