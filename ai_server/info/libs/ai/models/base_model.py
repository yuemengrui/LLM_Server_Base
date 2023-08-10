# *_*coding:utf-8 *_*
# @Author : YueMengRui
from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def get_embeddings(self, **kwargs):
        """
        return embeddings
        """

    @abstractmethod
    def lets_chat(self, **kwargs):
        """
        return answer history
        """

    @abstractmethod
    def lets_batch_chat(self, **kwargs):
        """
        batch chat
        """
