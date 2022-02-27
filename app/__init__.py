# -*-  utf-8 -*-

from flask import Flask
import os
import sys


def create_app():
    from app.generator_restful import generate_rest_bp as generate_api
    from app.generator_restful import generator_restplus
    app = Flask(__name__)
    app.register_blueprint(generate_api, url_prefix='/test')

    return app


if __name__ == "__main__":
    create_app()