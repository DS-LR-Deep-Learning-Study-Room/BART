from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

# Header

class Gender(StrEnum):
    male = "남성"
    female = "여성"

class ChatType(StrEnum):
    daily = "일상 대화"
    debate = "토론 대화"

class ChatTopic(StrEnum):
    personal = "개인 및 관계"
    wellness = "미용과 건강"
    commerce = "상거래(쇼핑)"
    knowledge = "시사/교육"
    food = "식음료"
    hobby = "여가 생활"
    work = "일과 직업"
    life = "주거와 생활"
    fest = "행사"

class _DialogueInfo(BaseModel):
    id: UUID = Field(alias="dialogueID")
    num_participants: int = Field(alias="numberOfParticipants")
    num_utterances: int = Field(alias="numberOfUtterances")
    num_turns: int = Field(alias="numberOfTurns")
    type: ChatType
    topic: ChatTopic

class _ParticipantInfo(BaseModel):
    id: str = Field(alias="participantID")
    gender: Gender
    age: str
    residential_province: str = Field(alias="residentialProvince")

class _Header(BaseModel):
    dialogue_info: _DialogueInfo = Field(alias="dialogueInfo")
    participants_info: list[_ParticipantInfo] = Field(alias="participantsInfo")

# Body

class Dialogue(BaseModel):
    utterance_id: str = Field(alias="utteranceID")
    turn_id: str = Field(alias="turnID")
    participant_id: str = Field(alias="participantID")
    date: str
    time: str
    utterance: str

class _Body(BaseModel):
    dialogues: list[Dialogue] = Field(alias="dialogue")
    summary: str
    
# Chat

class Chat(BaseModel):
    header: _Header
    body: _Body
    
    @property
    def dialogues(self) -> str:
        return " ".join(
            map(
                lambda d: f"{d.participant_id}::{d.utterance}",
                self.body.dialogues
            )
        )

class ChatData(BaseModel):
    num_chats: int = Field(default=0, alias="numberOfItems")
    chats: list[Chat] = Field(default=[], alias="data")
    
    @property
    def dialogues(self) -> list[str]:
        mapped_dialogues = map(lambda c: c.dialogues, self.chats)
        return list(mapped_dialogues)
    
    def summary(self, index: int) -> str:
        return self.chats[index].body.summary

    def uterrances(self, index: int) -> list[str]:
        return list(map(lambda d: d.utterance, self.chats[index].body.dialogues))
    
    def merge_(self, chat_data: "ChatData"):
        print(
            f"Merging {len(chat_data.chats)} datas to data of size {len(self.chats)}..."
        )
        self.num_chats += chat_data.num_chats
        self.chats.extend(chat_data.chats)

    def __getitem__(self, index: int) -> str:
        return self.dialogues[index]