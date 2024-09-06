from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template

import utils
import tasks

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1,
            temperature=0.0)[0]
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred


class PostgresQuestionPredictor(GPT4Predictor):
    categories = ['0', '1', '2', '3', '4']

    def inference(self, ex, prompt):
        prompt = Template(prompt + "\n\n質問文:\n{{text}}\n\n回答文:").render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1,
            temperature=0.0)[0]
        try:
            return int(response.strip())
        except:
            pass

        return 0
