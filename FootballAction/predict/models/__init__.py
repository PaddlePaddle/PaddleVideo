"""
model init regist
"""
from models.model import regist_model, get_model
import models.tsn.tsn as tsn
import models.audio.audio as audio
import models.bmn.bmn as bmn
import models.action.action as action
# regist models, sort by alphabet
regist_model("TSN", tsn.TSN)
regist_model("AUDIO", audio.AudioNet)
regist_model("BMN", bmn.BMN)
regist_model("ACTION", action.ActionNet)
