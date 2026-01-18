import yaml
import os

class PromptManager:
    _prompts = {}

    @classmethod
    def load(cls):
        path = os.path.join(os.getcwd(), "config/prompts.yaml")
        with open(path, "r") as f:
            cls._prompts = yaml.safe_load(f)

    @classmethod
    def get(cls, name):
        if not cls._prompts: cls.load()
        return cls._prompts.get(name, {}).get("template", "")