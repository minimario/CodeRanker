from run_seq_classification_partial import reindent_code


def preprocess_function(tokenizer, max_seq_length, examples):
    problems = [reindent_code(s) for s in examples["problem"]]
    solutions = [reindent_code(s) for s in examples["code"]]

    truncate_texta_from_first = False
    mask_padding_with_zero = True
    pad_token = 0
    pad_token_segment_id = 0
    pad_on_left = False

    def trunc(tokens_a, tokens_b, max_length, truncate_texta_from_first=False):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                if truncate_texta_from_first:
                    tokens_a.pop(0)
                else:
                    tokens_a.pop()
            else:
                tokens_b.pop()

    def custom_tokenize(text1, text2):
        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        for i in range(len(text1)):
            tok_seq1 = tokenizer.tokenize(text1[i])
            tok_seq2 = tokenizer.tokenize(text2[i])

            trunc(
                tok_seq1,
                tok_seq2,
                max_seq_length - 3,
                truncate_texta_from_first=truncate_texta_from_first,
            )  # 3 is number of special tokens for bert sequence pair

            input_ids = [tokenizer.cls_token_id]
            input_ids += tokenizer.convert_tokens_to_ids(tok_seq1)
            input_ids += [tokenizer.sep_token_id]

            token_type_ids = [0] * len(input_ids)

            input_ids += tokenizer.convert_tokens_to_ids(tok_seq2)
            input_ids += [tokenizer.sep_token_id]
            token_type_ids += [1] * (len(tok_seq2) + 1)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([tokenizer.pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + (
                    [pad_token_segment_id] * padding_length
                )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)

        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "label": examples["label"],
        }
        return result

    return custom_tokenize(problems, solutions)
