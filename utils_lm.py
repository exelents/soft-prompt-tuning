from transformers.data.datasets.language_modeling import *
import copy


class LineByLineWebNLGTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
                 block_size: int, bos_tok:str, eos_tok: str,
                 n_prefix_tokens: int, id_prefix_token: [int, list]):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)
        self.n_prefix_tokens = n_prefix_tokens
        self.id_prefix_token = id_prefix_token

        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in sents:
                if sent["comment"] == 'good':
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)



        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)


        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True,
                                   max_length=block_size-n_prefix_tokens,
                                   is_split_into_words=False)
        # !!!!
        # БОЛВАНКА ПРЕФИКСА
        pad_token = tokenizer.pad_token_id
        if pad_token is None:
            pad_token = -100

        if isinstance(id_prefix_token, int):
            id_prefix_token = [id_prefix_token]
        self.examples = []
        self.labels = []
        for e in batch_encoding["input_ids"]:
            for pt_id in id_prefix_token:
                self.examples.append([pt_id] * n_prefix_tokens + e)
                self.labels.append([pad_token] * n_prefix_tokens + e)


        # self.labels = copy.deepcopy(self.examples)

        # split into category words:
        # ssl_lst = full_rela_lst
        #
        # self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
        #                          is_split_into_words=True)['input_ids']
        #
        # self.src_sent = []
        # self.tgt_sent = []
        # if True:
        #     separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        #     for i, elem in enumerate(self.labels):
        #         try:
        #             sep_idx = elem.index(separator) + 1
        #         except ValueError:
        #             self.labels[i] = None
        #             continue
        #         self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
        #         self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
        #         self.labels[i][:sep_idx] = [-100] * sep_idx
        #
        #     self.labels = [l for l in self.labels if l is not None]

        print(self.examples[0])
        print(self.labels[0])
        print()
        print(self.examples[1])
        print(self.labels[1])
        assert len(self.labels) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                )