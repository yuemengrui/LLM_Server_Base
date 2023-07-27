# *_*coding:utf-8 *_*
# @Author : YueMengRui
from flask import Blueprint

chat_blu = Blueprint('Chat', __name__)

from . import views
