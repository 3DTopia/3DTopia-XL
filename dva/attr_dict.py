# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json


class AttrDict:
    def __init__(self, entries):
        self.add_entries_(entries)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        return self.__dict__.__delitem__(key)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def __getattr__(self, attr):
        if attr.startswith("__"):
            return self.__getattribute__(attr)
        return self.__dict__[attr]

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return iter(self.items())

    def add_entries_(self, entries, overwrite=True):
        for key, value in entries.items():
            if key not in self.__dict__:
                if isinstance(value, dict):
                    self.__dict__[key] = AttrDict(value)
                else:
                    self.__dict__[key] = value
            else:
                if isinstance(value, dict):
                    self.__dict__[key].add_entries_(entries=value, overwrite=overwrite)
                elif overwrite or self.__dict__[key] is None:
                    self.__dict__[key] = value

    def serialize(self):
        return json.dumps(self, default=self.obj_to_dict, indent=4)

    def obj_to_dict(self, obj):
        return obj.__dict__

    def get(self, key, default=None):
        return self.__dict__.get(key, default)
