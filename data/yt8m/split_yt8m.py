#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import sys
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO

def split_yt8m(filelist):
    print(__file__, sys._getframe().f_lineno, "filelist:",filelist)
    fl = open(filelist).readlines()
    fl = [line.strip() for line in fl if line.strip() != '']
    for filepath in fl:
       data = pickle.load(open(filepath, 'rb'), encoding='bytes')
       save_pre = filepath[:-4]  #erase ".pkl"
       indexes = list(range(len(data)))
       print("filepath:",filepath, "len(data):", len(data), "save_pre:", save_pre)
       for i in indexes:
           save_path = save_pre + "_split_" + str(i) +".pkl"
           record = data[i]
           output = open(save_path, 'wb')
           pickle.dump(data[i], output, protocol=2)
           output.close()

if __name__ == "__main__":
    split_yt8m("data/val.list")
