"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import os
import time
import requests
import string
from datetime import datetime


def setup_logger():
    from logging import getLogger, Formatter, NullHandler, FileHandler, INFO
    logger = getLogger(__name__)
    logger.setLevel(INFO)

    logging_enabled = os.getenv('LOGGING_ENABLED')

    if logging_enabled:
        fn = os.path.basename(__file__).split('.')[0]
        log_file = '.prompt_optimization_{fn}_{ts}.log'.format(fn=fn, ts=datetime.now().strftime('%Y%m%d_%H%M%S'))
        file_handler = FileHandler(log_file)
        format = "%(levelname)s  %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
        file_handler.setFormatter(Formatter(format))
        logger.addHandler(file_handler)
    else:
        logger.addHandler(NullHandler())

    return logger


# Logger for internal use in this module
_logger = setup_logger()


def escape_string(s):
    return s.replace('"', '\\"').replace('\n', '\\n')


def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def _is_incorrect_api_key(r):
    return r['error']['code'] == 'invalid_api_key'


def _is_rate_limit_exceeded(r):
    # See: https://platform.openai.com/docs/guides/rate-limits
    return r['error']['code'] == 'rate_limit_exceeded'


def _is_invalid_request(r):
    return r['error']['type'] == 'invalid_request_error'


def chatgpt_healthcheck():
    try:
        chatgpt('Hello!')
    except Exception as e:
        raise RuntimeError(f"Cannot access OpenAI API because: {str(e)}")

    pass


def _get_openai_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise Exception('No OpenAI API key found')

    return api_key


def _get_openai_api_host():
    api_host = os.getenv('OPENAI_API_HOST')
    if api_host is None:
        return 'https://api.openai.com/v1/chat/completions'

    return api_host


def _get_openai_model_name():
    model_name = os.getenv('OPENAI_MODEL_NAME')
    if model_name is None:
        return 'gpt-4o-mini'

    return model_name


def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024,
            presence_penalty=0, frequency_penalty=0, logit_bias={}, connection_timeout=3, read_timeout=300,
            max_retries=8, retry_interval=3):
    api_host = _get_openai_api_host()
    model_name = _get_openai_model_name()
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    }

    retries = 0
    while True:
        if retries >= max_retries:
            raise RuntimeError(f"Max retries (max_retries={max_retries}) exceeded")

        try:
            r = requests.post(api_host,
                headers = {
                    "Authorization": f"Bearer {_get_openai_key()}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=(connection_timeout, read_timeout)
            )
            if r.status_code != 200:
                if _is_incorrect_api_key(r.json()):
                    raise RuntimeError("Incorrect API key provided")
                if _is_rate_limit_exceeded(r.json()):
                    raise RuntimeError("Rate limit exceeded")
                if _is_invalid_request(r.json()):
                    raise RuntimeError(f"Invalid request: {r.json()}")

                _logger.warning(f"status_code={r.status_code}, retries={retries}, error={r.json()}")

                retries += 1
                time.sleep(retry_interval)
            else:
                break
        except requests.exceptions.ReadTimeout as e:
            _logger.warning(f"timeout, error={str(e)}")
            time.sleep(retry_interval)
            retries += 1

    r = r.json()
    res = [choice['message']['content'] for choice in r['choices']]

    _logger.info("payload={}, response={}".format(payload, res))

    return res

def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                headers = {
                    "Authorization": f"Bearer {_get_openai_key()}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=10
            )
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r['choices']
