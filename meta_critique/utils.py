import asyncio
import json
import os

import openai
import tiktoken
from tqdm import tqdm


def read_json(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data


def write_json(data, json_file_path):
    if os.path.dirname(json_file_path) != "" and not os.path.exists(
        os.path.dirname(json_file_path)
    ):
        os.makedirs(os.path.dirname(json_file_path))
    with open(json_file_path, "w") as f:
        json.dump(data, f)
    f.close()


def cls_post_process(out):
    if not isinstance(out, str):
        return None

    formatted_out = out.strip().lower()
    if formatted_out == "failed!":
        return None

    if "claim is true" in formatted_out:
        return {"explanation": out, "verifying_result": True}
    elif "claim is false" in formatted_out:
        return {"explanation": out, "verifying_result": False}
    else:
        return {"explanation": out, "verifying_result": None}


def text_post_process(out):
    if not isinstance(out, str):
        return None

    formatted_out = out.strip().lower()
    if formatted_out == "failed!":
        return None
    else:
        return {"output": out}


def generate_outputs(
    data_inputs, batched_openai_engine, cache_outputs, batch_size=3, cls_flag=False
):
    cost_total = 0
    if os.path.exists(cache_outputs):
        data_outputs = read_json(cache_outputs)
    else:
        data_outputs = [None for _ in data_inputs]

    for _ in range(5):
        # try 5 time to generate data
        new_all_data = []
        new_all_data_idx = []
        cur_idx = 0
        for cur_input, cur_output in zip(data_inputs, data_outputs):
            if cur_output is None:
                new_all_data.append(cur_input)
                new_all_data_idx.append(cur_idx)
            cur_idx += 1

        if len(new_all_data) == 0:
            break
        print("current processing number:", len(new_all_data_idx))
        # batch generation
        for batch_start in range(0, len(new_all_data), batch_size):
            batch_end = min(batch_start + batch_size, len(new_all_data))
            batched_openai_inputs = new_all_data[batch_start:batch_end]
            batched_output = batched_openai_engine.generate_batch(
                batched_openai_inputs, enable_tqdm=True
            )
            for openai_raw_output, cur_idx in zip(
                batched_output, new_all_data_idx[batch_start:batch_end]
            ):
                pred, cost = openai_raw_output
                cost_total += cost
                if cls_flag:
                    data_outputs[cur_idx] = cls_post_process(pred)
                else:
                    data_outputs[cur_idx] = text_post_process(pred)
            write_json(data_outputs, cache_outputs)

            print(f"batch: {batch_start} to {batch_end} finished")
            print("Current total cost is", cost_total)
    return cost_total, data_outputs


def build_message(sys_msg, task_input):
    return {"sysmsg": sys_msg, "usermsg": task_input}


def num_tokens_from_string(string, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_txt(txt_file_path):
    if txt_file_path is None:
        return None
    with open(txt_file_path, "r") as f:
        lines = f.readlines()
        return "".join(lines)


def calculate_cost(usage, model_name):
    """
    This function is used to calculate the cost of a request.
    :param usage:
    :param model_name:
    :return:
    """
    mapping = {
        "gpt-3.5-turbo": (0.0015, 0.002),
        "gpt-3.5-turbo-0613": (0.001, 0.002),
        "gpt-3.5-turbo-1106": (0.001, 0.002),
        "gpt-3.5-turbo-16k": (0.003, 0.004),
        "gpt-4": (0.03, 0.06),
        "gpt-4-0613": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
        "gpt-4-1106-preview": (0.06, 0.12),
    }
    intokens = usage.prompt_tokens
    outtokens = usage.completion_tokens

    assert model_name in mapping.keys()
    return (
        mapping[model_name][0] * intokens / 1000
        + mapping[model_name][1] * outtokens / 1000
    )


class OpenAIChat:
    """
    This class is a more complex wrapper for OpenAI API, support async batch generation.
    """

    def __init__(
        self,
        api_key=None,
        org_id=None,
        api_base=None,
        model="gpt-3.5-turbo",
        seed=None,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=60,
    ):
        # self.max_length = 16385
        self.max_length = 4096
        self.config = {
            "model_name": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "request_timeout": request_timeout,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if seed is not None:
            self.config["seed"] = seed

        openai.api_key = api_key
        if org_id is not None:
            openai.organization = org_id
        if api_base is not None:
            openai.api_base = api_base

    async def dispatch_openai_requests(self, messages_list, enable_tqdm):
        """Dispatches requests to OpenAI API asynchronously.

        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """

        async def _request_with_retry(id, messages, retry=3):
            for _ in range(retry):
                try:
                    if "seed" in self.config:
                        response = await openai.ChatCompletion.acreate(
                            model=self.config["model_name"],
                            messages=messages,
                            max_tokens=min(
                                self.config["max_tokens"],
                                self.max_length
                                - 100
                                - sum(
                                    [
                                        num_tokens_from_string(m["content"])
                                        for m in messages
                                    ]
                                ),
                            ),
                            temperature=self.config["temperature"],
                            top_p=self.config["top_p"],
                            seed=self.config["seed"],
                            request_timeout=self.config["request_timeout"],
                            frequency_penalty=self.config["frequency_penalty"],
                            presence_penalty=self.config["presence_penalty"],
                        )
                    else:
                        response = await openai.ChatCompletion.acreate(
                            model=self.config["model_name"],
                            messages=messages,
                            max_tokens=min(
                                self.config["max_tokens"],
                                self.max_length
                                - 100
                                - sum(
                                    [
                                        num_tokens_from_string(m["content"])
                                        for m in messages
                                    ]
                                ),
                            ),
                            temperature=self.config["temperature"],
                            top_p=self.config["top_p"],
                            request_timeout=self.config["request_timeout"],
                            frequency_penalty=self.config["frequency_penalty"],
                            presence_penalty=self.config["presence_penalty"],
                        )
                    return id, response
                except openai.error.RateLimitError:
                    print("Rate limit error, waiting for 40 second...")
                    await asyncio.sleep(40)
                except openai.error.APIError:
                    print("API error, waiting for 1 second...")
                    await asyncio.sleep(1)
                except openai.error.Timeout:
                    print("Timeout error, waiting for 1 second...")
                    await asyncio.sleep(1)
                except openai.error.ServiceUnavailableError:
                    print("Service unavailable error, waiting for 3 second...")
                    await asyncio.sleep(3)
            return id, None

        async def _dispatch_with_progress():
            async_responses = [
                _request_with_retry(index, messages)
                for index, messages in enumerate(messages_list)
            ]
            if enable_tqdm:
                pbar = tqdm(total=len(async_responses))
            tasks = asyncio.as_completed(async_responses)

            responses = []

            for task in tasks:
                index, response = await task
                if enable_tqdm:
                    pbar.update(1)
                responses.append((index, response))

            if enable_tqdm:
                pbar.close()

            responses.sort(key=lambda x: x[0])  # 根据索引排序结果

            return [response for _, response in responses]

        return await _dispatch_with_progress()

    async def async_run(self, messages_list, enable_tqdm):
        retry = 1
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            # print(f'{retry} retry left...')
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]

            predictions = await self.dispatch_openai_requests(
                messages_list=messages_list_cur, enable_tqdm=enable_tqdm
            )

            preds = [
                (
                    prediction["choices"][0]["message"]["content"],
                    calculate_cost(prediction["usage"], self.config["model_name"]),
                )
                if prediction is not None
                else ("Failed!", 0.0)
                for prediction in predictions
            ]

            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])

            messages_list_cur_index = [
                i for i in messages_list_cur_index if i not in finised_index
            ]

            retry -= 1

        return responses

    def generate_batch(self, msgs, enable_tqdm):
        """
        :param msgs: be like [{"sysmsg":"xx","usermsg":"yy"},...]
        :return:
        """
        msg_list = [
            [
                {"role": "system", "content": msg_pair["sysmsg"]},
                {"role": "user", "content": msg_pair["usermsg"]},
            ]
            for msg_pair in msgs
        ]
        predictions = asyncio.run(
            self.async_run(messages_list=msg_list, enable_tqdm=enable_tqdm)
        )
        # each prediction is a tuple (response, cost)
        return predictions

    def generate_single(self, msg):
        """
        this is just a wrapper for generate_batch when only one msg is given
        :param msg: be like {"sysmsg":"xx","usermsg":"yy"}
        :return:
        """
        msg_list = [
            [
                {"role": "system", "content": msg["sysmsg"]},
                {"role": "user", "content": msg["usermsg"]},
            ]
        ]
        predictions = asyncio.run(
            self.async_run(messages_list=msg_list, enable_tqdm=False)
        )
        return predictions[0]
