import json
import os
import re
import time

import requests
from rich import print

from .key_config import load_key_config

# Set HuggingFace token from environment or config file
cfg = load_key_config()
access_token = os.environ.get("HF_TOKEN")
if (access_token is None or access_token == "") and cfg:
    access_token = cfg.get("HF_TOKEN")
assert(access_token)
HF_TOKEN = access_token
hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Set HuggingFace Endpoint URL from environment or config file
endpoint = os.environ.get("HF_API_URL")
if (endpoint is None or endpoint == "") and cfg:
    endpoint = cfg.get("HF_API_URL")
assert(endpoint != None)
HF_API_URL = endpoint


def HFChat(message):
    # Prevent RateLimit errors
    time.sleep(1)
    payload = {"inputs": message, "parameters": {"max_new_tokens": 100, "temperature": 0.01}}
    response = requests.post(HF_API_URL, headers=hf_headers, json=payload)
    result = ""
    output = response.json()
    if len(output) != 0:
        result = output[0]["generated_text"].strip()
        re.sub("\n", " ", result)
        split_result = result.split("<|end|>")
        if len(split_result) > 1:
            result = split_result[0]
        split_result = result.split("<|assistant|>")
        if len(split_result) > 1:
            result = split_result[1]
        split_result = result.split("<|system|>")
        if len(split_result) > 1:
            result = split_result[0]
        split_result = result.split()
        if len(split_result) > 1 and split_result[0] == "sh":
            result = " ".join(result.split()[1:])
    return result.strip()
