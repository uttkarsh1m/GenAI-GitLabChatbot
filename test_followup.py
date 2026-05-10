from src.chatbot import GitLabChatbot

bot = GitLabChatbot()
bot.conversation_history = [{"role": "user", "content": "test"}]
print("Is followup?", bot.is_followup("why"))
