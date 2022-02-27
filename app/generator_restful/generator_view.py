#! /usr/bin/env python
# -*- coding: utf-8

from flask_restx import Resource
from app.generator_restful import generator_restplus as api
from . import generator_model


@api.route("/hello")
class Generator(Resource):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.model = generator_model

    def get(self):
        return self.model.detect()

    def post(self):
        sentence = self.api.payload.get("sentence", None)
        num = int(self.api.payload.get("num", 10))
        return self.model.detect(sentence=sentence, num=num)

