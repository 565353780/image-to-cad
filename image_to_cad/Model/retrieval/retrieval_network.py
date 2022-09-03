#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import torch.nn as nn

class RetrievalNetwork(nn.Module, abc.ABC):
    @abc.abstractproperty
    def embedding_dim(self):
        pass

