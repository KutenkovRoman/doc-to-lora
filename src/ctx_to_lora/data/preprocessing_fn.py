import logging
import random
from collections.abc import Callable
from typing import Any

from ctx_to_lora.data.definitions import CLOSED_QA_INTX_TEMPLATES, EVAL_INTX_TEMPLATES
from ctx_to_lora.utils import concat_list

logger = logging.getLogger()


def closed_qa_prompting(prompt: str):
    template = random.choice(CLOSED_QA_INTX_TEMPLATES)
    return template.format(input=prompt)


def chat_to_str(messages: list[dict[str, str]]):
    return "Below is the chat history from the current user.\n\n" + "\n\n".join(
        [
            "Message from: {role}\n{content}".format(
                **{**m, "role": m["role"].capitalize()}
            )
            for m in messages
        ]
    )


def get_preprocessing_fn(
    ds_name: str,
    is_eval: bool,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Get preprocessing function for a specific dataset.

    Args:
        ds_name: Name of the dataset

    Returns:
        A preprocessing function that takes and returns a dictionary
    """
    f = lambda x: x
    if ds_name.startswith("self_gen") or ds_name.endswith("_compact"):
        # already processed data, do nothing
        return f

    if "fw_qa_v2" in ds_name:

        def f(sample):
            # get questions/answers from all levels in the ds
            q_cols = [col for col in sample.keys() if col.startswith("prompts_level")]
            r_cols = [col for col in sample.keys() if col.startswith("responses_level")]
            questions = concat_list([sample[col] for col in q_cols])
            responses = concat_list([sample[col] for col in r_cols])
            min_len = min(len(questions), len(responses))

            if min_len == 0:
                return {
                    "context": None,
                    "prompts": None,
                    "responses": None,
                }

            return {
                "context": sample["context"],
                "prompts": questions[:min_len],
                "responses": responses[:min_len],
            }

    elif ds_name.startswith("longbench"):

        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["input"],
                "response": sample["answers"][0],
            }

    elif ds_name == "pwc" or ds_name == "pwc_tiny":
        # original pwc
        def f(sample):
            return {
                "context": sample["input"],
                "prompt": sample["prompt"],
                "response": sample["answer"],
            }

    elif ds_name == "squad":
        # original squad
        def f(sample):
            q = sample["question"]
            prompt = closed_qa_prompting(q) if not is_eval else q
            return {
                "context": sample["context"],
                "prompt": prompt,
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "squad_assistant_ctx":

        def f(sample):
            return {
                "context": "You are a useful AI assistant.",
                "prompt": sample["context"] + "\n\n" + sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "squad_negative":
        with open("data/gutenburg_sample.txt") as f:
            gutenburg_sample = f.read()

        def f(sample):
            return {
                "context": gutenburg_sample,
                "prompt": sample["context"] + "\n\n" + sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "squad_negative_no_passage":
        with open("data/gutenburg_sample.txt") as f:
            gutenburg_sample = f.read()

        def f(sample):
            return {
                "context": gutenburg_sample,
                "prompt": sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "squad_assistant_ctx_no_passage":

        def f(sample):
            return {
                "context": "You are a useful AI assistant.",
                "prompt": sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "drop":

        def f(sample):
            q = sample["question"]
            prompt = closed_qa_prompting(q) if not is_eval else q
            return {
                "context": sample["passage"],
                "prompt": prompt,
                "response": sample["answers_spans"]["spans"][0],
            }

    elif ds_name == "ropes":
        ctx_template = "{background}\n{situation}"

        def f(sample):
            response = sample["answers"]["text"][0]
            bg_txt = sample["background"]
            situation_txt = sample["situation"]
            ctx = ctx_template.format(background=bg_txt, situation=situation_txt)
            q = sample["question"]
            q = closed_qa_prompting(q) if not is_eval else q
            return {"context": ctx, "prompt": q, "response": response}

    elif ds_name.startswith("babilong_qa1"):

        def f(sample):
            return {
                "context": sample["input"].replace("\n\n\n", " ").replace("\n\n", " ").replace("\n", " "),
                # For now prompt only has a question unlike in babilong repo where there are
                # additional examples and instruction on how to format answer
                "prompt": sample["question"] + (
                    "Use the latest location to answer the question. Output only the location and do not output any other words."
                    if not ds_name.endswith("external") else ""
                ),
                "response": sample["target"],
            }

    elif ds_name.startswith("babilong_qa2"):

        def f(sample):
            return {
                "context": sample["input"].replace("\n\n\n", " ").replace("\n\n", " ").replace("\n", " "),
                # For now prompt only has a question unlike in babilong repo where there are
                # additional examples and instruction on how to format answer
                "prompt": sample["question"] + (
                    "If a person got an item in the first location and travelled to the second location the item is also in the second location. If a person dropped an item in the first location and moved to the second location the item remains in the first location."
                    # " Output only the location and do not output any other words."
                    if not ds_name.endswith("external") else ""
                ),
                "response": sample["target"],
            }
    
    elif ds_name.startswith("ab_qa1"):

        def f(sample):
            return {
                "context": sample["input"].replace("\n", " "),
                "prompt": sample["question"] + "Output only the location and do not output any other words.",
                # "prompt": sample["question"][:-1],
                "response": sample["target"],
                # "response": ("Alice" if "Alice" in sample["question"] else "Bob") + " is in the " + sample["target"],
            }

    elif ds_name == "tool_mix_single_tool":
        HEADER = (
            "You are a helpful LLM agent that calls all necessary tools (usually, more than "
            "one in total) and uses the information from those tools to fulfill the user's "
            "request as accurately as possible. You are given a question and a set of possible "
            "functions.\nBased on the question, you will need to make one or more function/tool "
            "calls to achieve the purpose.\nYou should only return the function call in tool call "
            "sections.\nHere is a list of functions in JSON format that you can invoke:\n"
        )
        FOOTER = (
            "\nShould you decide to return the function call(s). Put it in the format of "
            "[func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n\n"
            "NO other text MUST be included."
        )

        def f(sample):
            return {
                "context": HEADER + sample["available_tools"] + FOOTER,
                "prompt": sample["prompt"],
                "response": sample["response"],
            }


    if is_eval and (ds_name in EVAL_INTX_TEMPLATES):
        prompt_template = EVAL_INTX_TEMPLATES[ds_name]

        def eval_intx_decorator(f):
            def g(sample):
                sample = f(sample)
                assert "prompt" in sample, (
                    f"Expected 'prompt' in sample, got {sample.keys()}"
                )
                sample["prompt"] = prompt_template.format(input=sample["prompt"])
                return sample

            return g

        f = eval_intx_decorator(f)

    def maybe_convert_to_list(f):
        def g(sample):
            sample = f(sample)
            if "prompt" in sample:
                sample["prompts"] = [sample.pop("prompt")]
            if "response" in sample:
                sample["responses"] = [sample.pop("response")]
            return sample

        return g

    f = maybe_convert_to_list(f)

    if "self_gen" not in ds_name:

        def strip_response(f):
            def g(sample):
                sample = f(sample)
                if "responses" in sample and bool(sample["responses"]):
                    sample["responses"] = [
                        r.strip() if isinstance(r, str) else r
                        for r in sample["responses"]
                    ]
                return sample

            return g

        f = strip_response(f)

    return f
