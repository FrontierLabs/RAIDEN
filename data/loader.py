# -*- coding: utf-8 -*-

"""
    Load character data
"""
import sys   
import importlib
importlib.reload(sys)

import os
import re
from dataclasses import dataclass
from typing import List, Dict

import json

from .generic import Role


@dataclass
class DataLoaderOutput:
    ID: str = None
    messages: List[Dict] = None
    reference: str = None
    metrics: str = None
    npc_type: str = None
    npc_name: str = None
    npc_setting: str = None


class DataLoader:
    def __init__(self, data_dir):
        self.npc, self.dialogue = self.load_data(data_dir)

    def load_data(self, data_dir):
        with open(os.path.join(data_dir, "npc.json")) as reader:
            npc = json.load(reader)

        with open(os.path.join(data_dir, "dialogue.json")) as reader:
        # with open(os.path.join(data_dir, "dialogue_test.json")) as reader:
            dialogue = json.load(reader)
        return npc, dialogue

    def __iter__(self) -> DataLoaderOutput:
        """Returns a set of data each time

        :return messages: List[dict], historical information
                Each message format is as follows:
                    {
                        "role": speaker, #  Role.USER / Role.ASSISTANT
                        "text": speech content
                    }
        :return reference: str, standard answer
        :return metrics: evaluation metrics
        :return npc_name: str character name
        :return npc_setting: str character setting
        :return npc_type: NPC character type
        """
        for ID in self.dialogue:
            dialogue = self.dialogue[ID]
            messages = dialogue["messages"]
            for message in messages:
                if message["role"] == "user":
                    message["role"] = Role.USER
                else:
                    message["role"] = Role.ASSISTANT

            npc_name = dialogue["npc_name"]
            output = DataLoaderOutput(messages=messages, reference=dialogue["reference"], metrics=dialogue["metrics"],
                                      npc_name=npc_name, npc_type=self.npc[npc_name]["npc_type"],
                                      npc_setting=self.npc[npc_name]["npc_setting"], ID=ID)

            yield output
