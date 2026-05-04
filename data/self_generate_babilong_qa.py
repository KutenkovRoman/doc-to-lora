import argparse
import os
import re

import numpy as np
import yaml
from datasets import Dataset  #load_dataset
from vllm import LLM, SamplingParams

import sys
sys.path.append("src")  # complains that there is no module ctx_to_lora without this

from ctx_to_lora.data.definitions import SELF_GEN_DATA_DIR
from ctx_to_lora.data.processing import (
    filter_none,
    load_and_process_dataset,
    tokenize_ctx_text,
)
from ctx_to_lora.data.self_gen_template import (
    SELF_GEN_SYSTEM_MSG,
    SELF_QA_INTX,
)
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.utils import clear_gpu


MODEL_CTX_LEN = {
    "google/gemma-2-27b-it": 8192,
    "google/gemma-2-2b-it": 8192,
    "google/gemma-2-9b-it": 8192,
    # qwen 4b has 256k ctx length but using lower max lengths is faster
    "Qwen/Qwen3-4B-Instruct-2507": 2**13 + 2**12,
}


def truncate_middle_if_too_long(
    input_ids: list[int],
    max_length: int,
    max_new_tokens: int = 256,
) -> list[int]:
    """
    Truncate the middle of a list of tokens to fit within a maximum length.

    Args:
        tokens: List of token IDs
        max_length: Maximum length for the truncated tokens

    Returns:
        List of truncated token IDs
    """
    max_new_tokens_half = max_new_tokens // 2
    # leave max_new_tokens for generation
    half = max_length // 2 - max_new_tokens_half
    if len(input_ids) > max_length:
        return input_ids[:half] + input_ids[-half:]
    return input_ids


def load_config(config_path: str) -> dict:
    """Load dataset names from YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_dataset_configs(
    ds_names: list[str] | None,
    config: dict | None,
    split: str | None,
) -> list[tuple[str, str]]:
    assert not (ds_names and config), "Cannot provide both ds_names and config"
    if ds_names:
        assert split, "When using ds_names, --split must be provided"
        # Validate ds_names format
        for ds_name in ds_names:
            if not isinstance(ds_name, str):
                raise ValueError(f"Invalid dataset name: {ds_name}")
        return [(ds_name, split) for ds_name in ds_names]

    if config:
        dataset_configs = []

        # Process train datasets
        train_ds_names = config.get("train_ds_names", [])
        # self_gen_train_ds_names = [
        #     (ds_name.split("/")[-1], "train")
        #     for ds_name in train_ds_names
        #     if ds_name.startswith("self_gen/")
        # ]
        self_gen_train_ds_names = [
            (ds_name, "train")
            for ds_name in train_ds_names
            if ds_name.startswith("self_gen/")
        ]
        if not self_gen_train_ds_names:
            print("No self_gen datasets found in train_ds_names")
        dataset_configs.extend(self_gen_train_ds_names)

        # Process validation datasets
        val_ds_names = config.get("val_ds_names", [])
        self_gen_val_ds_names = [
            (ds_name, "validation")
            for ds_name in val_ds_names
            if ds_name.startswith("self_gen/")
        ]
        if not self_gen_val_ds_names:
            print("No self_gen datasets found in val_ds_names")
        dataset_configs.extend(self_gen_val_ds_names)

        return dataset_configs


def self_generate(
    ds_name: str,
    split: str,
    args: argparse.Namespace,
    llm: LLM,
    system_template: str,
) -> None:
    """Process a single dataset and generate QA pairs."""

    shard_name = ""

    # Conflict checks for ds_name-derived overrides
    if ds_name is not None:
        # temperature & closed_qa already handled later; add new ones
        if "_temp_" in ds_name and args.temp != 0.0:
            raise ValueError(
                f"Multiple sources of truth for temperature: CLI arg --temp={args.temp} and dataset name contains temp specification."
            )

    # Base values from args (ditched closed_qa_prob parameter)
    temp = args.temp

    # Overrides from ds_name pattern if present
    if ds_name is not None:
        if "_temp_" in ds_name:
            m = re.search(r"_temp_([\d.]+)", ds_name)
            if m:
                temp = float(m.group(1))

    print(f"Processing dataset: {ds_name}, split: {split}")
    print(f"Using temperature: {temp}")

    ds_name = ds_name.split("/")[-1]  # Extract just the dataset name

    print(f"Loading dataset: {ds_name} with split: {split}")

    num_proc = 4
    # At this point we have columns: 'context', 'prompts' (with appropriate intx), 'responses'
    ds = load_and_process_dataset(
        ds_name,
        split,
        is_eval=True,  # To add instruction
        add_negative_prompt=False,
        num_proc=num_proc,
    )
    print(f"Loaded dataset: {ds_name} with split: {split}")

    ds = ds.filter(filter_none, batched=False, num_proc=num_proc)

    tokenizer = get_tokenizer(args.vllm_model, train=True)

    self_qa_intx_tokens = tokenizer(" \n\n", add_special_tokens=False)["input_ids"]

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # After this we have new column 'ctx_ids' with tokenized contexts
    ds = ds.map(
        tokenize_ctx_text,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=50_000,
        keep_in_memory=True,
    )

    # ctxs = [sample["context"] for sample in ds]
    # questions = [q_list for q_list in ds["prompts"] if len(q_list) > 0]
    ctxs, questions = ds["context"], ds["prompts"]

    print(f"Loaded {len(ctxs)} contexts and {len(questions)} questions")

    # fpath = f"{SELF_GEN_DATA_DIR}/qwen3-4b-instruct/{ds_name}/{split}/ds{shard_name}"
    fpath = f"{SELF_GEN_DATA_DIR}/gemma-2-2b-it/{ds_name}/{split}/ds{shard_name}"
    chunk_size = 1_000

    for chunk_idx, start in enumerate(range(0, len(ctxs), chunk_size)):
        print(f"Processing chunk {chunk_idx}")

        chunk_ctxs = ctxs[start:(start + chunk_size)]
        chunk_questions = questions[start:(start + chunk_size)]

        chunk_ctx_ids = ds[start:(start + chunk_size)]["ctx_ids"]
        chunk_answers = ds[start:(start + chunk_size)]["responses"]

        chunk_messages = [
            [
                {
                    "role": "user",
                    # Qwen tokenizer parses ".\n\n" as a separate token, probably does
                    # same shit with other combos...
                    "content": (ctx.rstrip() + " \n\n" + q).strip(),
                }
            ]
            for ctx, q_list in zip(chunk_ctxs, chunk_questions)
            for q in q_list  # for babilong q_list always has 1 element
        ]

        print(f"Generating from {len(chunk_messages)} contexts")

        # Clear GPU memory before processing the next chunk
        clear_gpu()
        execute_qa_generation(
            fpath + f"_{chunk_idx:04d}",
            args,
            llm,
            temp,
            tokenizer,
            self_qa_intx_tokens,
            chunk_ctxs,
            chunk_ctx_ids,
            chunk_answers,
            chunk_messages,
        )


def execute_qa_generation(
    fpath,
    args,
    llm,
    temp,
    tokenizer,
    self_qa_intx_tokens,
    ctxs,
    ctx_ids,
    answers,
    messages,
):
    valid_answers = [
        "bathroom", "bedroom", "garden", "hallway", "kitchen", "office"
    ]
    k = len(valid_answers)
    n_self_qa_intx_tokens = len(self_qa_intx_tokens)

    completions = llm.chat(
        messages,
        sampling_params=SamplingParams(
            max_tokens=args.max_new_tokens,
            logprobs=k,
            temperature=temp,
            seed=42,
            spaces_between_special_tokens=False,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        ),
    )

    self_gen_data = {
        ctx: {
            "ctx_ids": ctx_ids,
            "input_ids": [],
            "response_start_end": [],
            "logprobs_vals": [],
            "logprobs_indices": [],
        }
        for ctx, ctx_ids in zip(ctxs, ctx_ids)
    }
    c = 0
    n_skips = 0

    # qwen tokenizer encodes differently depending on position in a sequence, but I tested
    # and it should be fine...
    # offset by 1 for gemma tokenizer
    valid_answer_tokens = [tokenizer.encode(answer)[1:] for answer in valid_answers]
    valid_answer_start_tokens = [tokens[0] for tokens in valid_answer_tokens]

    _valid_answer_tokens = [tokenizer.encode(answer[0].upper() + answer[1:])[1:] for answer in valid_answers]
    _valid_answer_start_tokens = [tokens[0] for tokens in _valid_answer_tokens]

    for answer, tokens in zip(valid_answers, valid_answer_tokens):
        print(answer, '->', tokens)

    # MONKEY PATCH
    assert k == 6 and all(len(tokens) <= 2 for tokens in valid_answer_tokens)

    # valid_start_permutation = [  # input ids for qwen
    #     [65, 2721, 70, 42241, 74, 26516],  # bathroom
    #     [2721, 70, 42241, 74, 26516, 65],  # bedroom
    #     [70, 42241, 74, 26516, 65, 2721],  # garden
    #     [42241, 74, 26516, 65, 2721, 70],  # hallway
    #     [74, 26516, 65, 2721, 70, 42241],  # kitchen
    #     [26516, 65, 2721, 70, 42241, 74],  # office
    # ]
    # filler_logprob_indices = [  # input ids for qwen
    #     [77832,  587,  43561, 17169,  6207, 24736],  # bathroom
    #     [2966,   9750, 10721,  8719,  4191, 50670],  # bedroom
    #     [8341,  23118,  567,  45296, 14556, 13551],  # garden
    #     [3117,  41803,  2284, 26798,  1616, 24723],  # hallway
    #     [7454,  48621, 28324,  9780,  1610,  9400],  # kitchen
    #     # office does not need it since it is a 1 token
    #     [151645, 11162, 334, 198, 25521, 13],  # <|im_end|>
    # ]

    valid_start_permutation = [  # input ids for gemma
        [110277,  62083,  49186,  16810,  56745,  26878],  # bathroom
        [ 62083,  49186,  16810,  56745,  26878, 110277],  # bedroom
        [ 49186,  16810,  56745,  26878, 110277,  62083],  # garden
        [ 16810,  56745,  26878, 110277,  62083,  49186],  # hallway
        [ 56745,  26878, 110277,  62083,  49186,  16810],  # kitchen
        [ 26878, 110277,  62083,  49186,  16810,  56745],  # office
    ]
    filler_logprob_indices = [  # input ids for gemma
        # this time only hallway consists of 2 tokens...
        [235248, 107, 1, 108, 139, 235265],  # bathroom
        [235248, 107, 1, 108, 139, 235265],  # bedroom
        [235248, 107, 1, 108, 139, 235265],  # garden
        [1677, 65041, 51686, 2095, 108, 235248],  # hallway
        [235248, 107, 1, 108, 139, 235265],  # kitchen
        [235248, 107, 1, 108, 139, 235265],  # office
    ]

    cnt = 0
    uppercase = 0
    for ctx, a_list in zip(ctxs, answers):
        for i, answer in enumerate(a_list):
            reason = completions[c + i].outputs[0].finish_reason
            if reason != "stop":
                print(f"finish_reason: {completions[c + i].outputs[0].finish_reason}")
                print(f"Skipping due to finish_reason={reason} != 'stop'")
                n_skips += 1
                continue

            # includes the logprob before the first response token
            # but excludes the logprob from eos token
            logp = completions[c + i].outputs[0].logprobs

            # len = num response tokens
            n_response_tokens = len(completions[c + i].outputs[0].token_ids)

            logp_indices = np.empty((n_response_tokens, k), dtype=np.int32)
            # float-16 is better for this range
            logp_vals = np.empty((n_response_tokens, k), dtype=np.float16)
            assert len(logp) == n_response_tokens, (
                f"Expected {n_response_tokens} logp entries, got {len(logp)}"
            )

            for li, info_d in enumerate(logp):
                for j, (idx, tok_info) in enumerate(info_d.items()):
                    logp_indices[li, j] = idx
                    logp_vals[li, j] = tok_info.logprob

            prompt_ids = completions[c + i].prompt_token_ids  # 1d list
            # token_ids only includes generated tokens, not the prompt
            response_token_ids = completions[c + i].outputs[0].token_ids  # 1d list
            all_ids = prompt_ids + response_token_ids
            res_start = len(prompt_ids)
            res_end = res_start + n_response_tokens

            q_start = None
            for j in range(len(prompt_ids) - n_self_qa_intx_tokens, -1, -1):
                if prompt_ids[j:(j + n_self_qa_intx_tokens)] == self_qa_intx_tokens:
                    # found the start of the user input
                    q_start = j + n_self_qa_intx_tokens
                    break

            # For debug; will crush anyway if it activates
            if q_start is None:
                print(prompt_ids)

            # bos + question + eos + start model turn + response + eos
            input_ids = all_ids[q_start:res_end]

            # relative to the input_ids
            res_start = res_start - q_start
            res_end = res_start + n_response_tokens

            # arrays will be saved as nested lists of numbers

            answer_idx = valid_answers.index(answer)
            suffix_len = 3  # 1 for qwen

            if input_ids[res_start] in _valid_answer_start_tokens:
                # if uppercase < 3:
                #     print(tokenizer.decode(input_ids))
                #     print(f"start={res_start}, end={res_end}")
                #     print(f"response={tokenizer.decode(input_ids[res_start:res_end])}")
                #     for j in range(len(logp_vals)):
                #         print(f"logprob_vals[{j}]={logp_vals[j]}")
                #         print(f"logprobs_indices[{j}]={logp_indices[j]}")
                #         print("=" * 80)

                token = valid_answer_start_tokens[
                    _valid_answer_start_tokens.index(input_ids[res_start])
                ]
                input_ids[res_start] = token

                if token in logp_indices[0]:
                    # swap if needed
                    logp_indices[0, logp_indices[0] == token] = logp_indices[0, 0]
                logp_indices[0, 0] = token

                uppercase += 1

            # redistribute probability if the most probable response is not a valid answer
            # and (kind of) force true answer
            # if not any(
            #     input_ids[res_start:(res_end - 1)] == sequence
            #     for sequence in valid_answer_tokens
            # ):
            if input_ids[res_start:(res_end - suffix_len)] != valid_answer_tokens[answer_idx]:
                cnt += 1
                epsilon = 1e-9  # rest of probability mass (except for valid answers)
                input_ids = (
                    input_ids[:res_start] +
                    valid_answer_tokens[answer_idx] +
                    input_ids[(res_end - suffix_len):res_end]
                )
                logp_vals = np.array(
                    # probability for first token is uniform across valid answers
                    # [[np.log((1 - epsilon) / k)] * k] +
                    [
                        [  # twice as much probability for true answer
                            np.log((2 if j == 0 else 1) * (1 - epsilon) / (k + 1))
                            for j in range(k)
                        ]
                    ] + (
                        # enforcement of true answer + eos token
                        [
                            [
                                np.log(1 - epsilon if j == 0 else epsilon / (k - 1))
                                for j in range(k)
                            ]
                        ] * (len(valid_answer_tokens[answer_idx]) + suffix_len - 1)
                    ),
                    dtype=np.float16
                )
                logp_indices = np.concatenate(
                    (
                        np.array(
                            [valid_start_permutation[answer_idx]] + (
                                [filler_logprob_indices[answer_idx]]
                                # if answer_idx != 5  # answer is not "office" for qwen
                                if answer_idx == 3  # answer is "hallway" for gemma
                                else []
                            ),
                            dtype=np.int32
                        ),
                        logp_indices[-suffix_len:, :],
                    ),
                    axis=0,
                )
                res_end = res_start + len(valid_answer_tokens[answer_idx]) + suffix_len

            self_gen_data[ctx]["input_ids"].append(input_ids)
            # assume single-turn chat
            self_gen_data[ctx]["response_start_end"].append((res_start, res_end))
            self_gen_data[ctx]["logprobs_vals"].append(logp_vals)
            self_gen_data[ctx]["logprobs_indices"].append(logp_indices)

        c += i + 1

    print(f"Skipped {n_skips} responses due to missing stop strings")
    print(f"Modified {cnt} samples with {uppercase} being in upper case")
    samples = [
        {
            "ctx_ids": self_gen_data[ctx]["ctx_ids"],
            "input_ids": self_gen_data[ctx]["input_ids"],
            "response_start_end": self_gen_data[ctx]["response_start_end"],
            "logprobs_vals": self_gen_data[ctx]["logprobs_vals"],
            "logprobs_indices": self_gen_data[ctx]["logprobs_indices"],
        }
        for ctx in ctxs
    ]

    # stats = {token: 0 for token in valid_answer_start_tokens}
    cnt = 0
    print("=" * 80)
    for i, sample in enumerate(samples):
        if args.debug:
            print(f"QA={[tokenizer.decode(ids) for ids in sample['input_ids']]}")

            for input_ids, (start, end) in zip(
                sample["input_ids"], sample["response_start_end"]
            ):
                print(f"start={start}, end={end}, tokens=[{input_ids[start]}, {input_ids[start + 1]}, ...]")
                print(f"response={tokenizer.decode(input_ids[start:end])}")

            print(f"logprobs_vals={[x.shape for x in sample['logprobs_vals']]}")
            print(f"logprobs_indices={[x.shape for x in sample['logprobs_indices']]}")
            for logprobs, indices in zip(
                sample["logprobs_vals"], sample["logprobs_indices"]
            ):
                for i in range(len(logprobs)):
                    print(f"logprob_vals[{i}]={logprobs[i]}")
                    print(f"logprobs_indices[{i}]={indices[i]}")
                print("=" * 80)

        for input_ids, (start, end), logprobs, indices in zip(
            sample["input_ids"], sample["response_start_end"],
            sample["logprobs_vals"], sample["logprobs_indices"]
        ):
            if input_ids[start] not in valid_answer_start_tokens:  #and stats[input_ids[start]] < 3
                # stats[input_ids[start]] += 1
                cnt += 1
                print(f"response={tokenizer.decode(input_ids[start:end])}")
                print(f"logprob_vals[-2]={logprobs[-2]}")
                print(f"logprobs_indices[-2]={indices[-2]}")
                print(f"logprob_vals[-1]={logprobs[-1]}")
                print(f"logprobs_indices[-1]={indices[-1]}")
                print("=" * 80)

    print(f"Incorrect answers: {cnt}")
    print(f"Generated {len(samples)} samples")

    # Save results
    ds_out = Dataset.from_list(samples)

    if args.debug:
        fpath += "_debug"
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    fpath = f"{fpath}.parquet"
    ds_out.to_parquet(fpath)
    print(f"Saved to {fpath}")

    # Cleanup
    del samples, ds_out, completions, messages, ctxs, answers
    clear_gpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA pairs using VLLM")
    parser.add_argument(
        "--vllm_model",
        type=str,
        required=True,
        help="VLLM model name (e.g., google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process only 10 samples)",
    )

    # Either config file OR ds_names + split
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file with train_ds_names/val_ds_names",
    )
    group.add_argument(
        "--ds_names",
        type=str,
        nargs="+",
        help="List of dataset names/shard patterns",
    )
    group.add_argument(
        "--glob_pattern",
        type=str,
        help="Glob pattern to match dataset names (e.g., 'data/raw_datasets/fw_qa_3/*')",
    )

    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split to use when using --ds_names (required with --ds_names)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0)",
    )
    parser.add_argument(
        "--closed_qa_prob",
        type=float,
        default=0.0,
        help="Probability of using closed QA prompt template (default: 0.0)",
    )
    # parser.add_argument(
    #     "--do_truncate",
    #     action="store_true",
    #     help="Truncate contexts to fit model context length",
    # )
    parser.add_argument(
        "--remove_qa_template",
        action="store_true",
        help="Remove QA template formatting from prompts",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate arguments
    if args.ds_names and not args.split:
        raise ValueError("--split is required when using --ds_names")

    vllm_model = args.vllm_model
    print(f"Using model: {vllm_model}")

    # Setup model-specific configurations
    llm_kwargs = dict(
        model=vllm_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=MODEL_CTX_LEN.get(vllm_model),
        max_num_batched_tokens=16384,
        max_num_seqs=32,  # avoid oom when getting logprobs
    )

    print(f"{llm_kwargs=}")
    llm = LLM(**llm_kwargs)

    # Get dataset configs from config or CLI args
    config = load_config(args.config) if args.config else None

    dataset_configs = get_dataset_configs(
        ds_names=args.ds_names,
        config=config,
        split=args.split,
    )

    # Process each dataset
    for ds_name, split in dataset_configs:
        print(f"Processing dataset: {ds_name}, split: {split}")
        self_generate(ds_name, split, args, llm, SELF_GEN_SYSTEM_MSG)
