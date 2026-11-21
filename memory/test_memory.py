from memory.conversation import ConversationMemory

memory = ConversationMemory()

print("Before clearing:")
print(memory.get_history())

memory.clear_history()

print("\nAfter clearing:")
print(memory.get_history())