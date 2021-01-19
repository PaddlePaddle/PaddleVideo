"""
read map for model
"""
from reader.reader_utils import regist_reader, get_reader
import reader.tsninf_reader as tsninf_reader
import reader.audio_reader as audio_reader
import reader.bmninf_reader as bmninf_reader
import reader.feature_reader as feature_reader

# regist reader, sort by alphabet
regist_reader("TSN", tsninf_reader.TSNINFReader)
regist_reader("AUDIO", audio_reader.AudioReader)
regist_reader("BMN", bmninf_reader.BMNINFReader)
regist_reader("ACTION", feature_reader.FeatureReader)
