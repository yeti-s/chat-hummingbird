"""
A dedicated helper to manage templates and prompt building.
"""

import json
import fire
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str, verbose: bool = False):
        self._verbose = verbose
        file_name = osp.join(f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
            
    def get_num_inputs(self)->int:
        return self.template['num_inputs']

    def generate_prompt(
        self,
        inputs: dict[str, str],
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        
        res = self.template['prompt'].format(**inputs)
        
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    

def test_prompt(template_name:str):
    prompter = Prompter(template_name, True)
    
    inputs = {}
    for i in range(prompter.get_num_inputs()):
        inputs[f'input{i}'] = f'Prompt-Input-{i}'
    prompter.generate_prompt(inputs, 'Prompt Label')
    
if __name__ == "__main__":
    fire.Fire(test_prompt)