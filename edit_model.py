#!/usr/bin/env python
# -*- coding: utf-8 -*-

from image_to_cad.Method.model import changeModelName

if __name__ == "__main__":
    model_path = "./Models/model_best.pth"
    new_model_path = "./Models/model_best_1.pth"
    changeModelName(model_path, new_model_path)

