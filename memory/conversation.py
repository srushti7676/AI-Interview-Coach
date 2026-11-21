import json
import os


class ConversationMemory:

    def __init__(self, file_path="memory/chat_history.json"):
        self.file_path = file_path
        self.history = []

        self.load_history()

    def save_history(self):
        with open(self.file_path, "w") as file:
            json.dump(self.history, file, indent=4)

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                self.history = json.load(file)

    def add_message(self, role, content):
        self.history.append({
            "role": role,
            "content": content
        })

        self.save_history()

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []
        self.save_history()