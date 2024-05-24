from typing import Any

import telegram

from .secrets import TELEGRAM_API_KEY, TELEGRAM_CHAT_ID


class Messenger:
    def __init__(self):
        self.bot = telegram.Bot(token=TELEGRAM_API_KEY)

    async def send_message(self, message: Any):
        await self.bot.send_message(text=message, chat_id=TELEGRAM_CHAT_ID)
