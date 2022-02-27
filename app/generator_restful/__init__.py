from flask import Blueprint
from flask_restx import Api
from app.generator.detect import GeneratorModel
import sys

generate_rest_bp = Blueprint("main", __name__)
generator_restplus = Api(generate_rest_bp)

sys.path.insert(0, "./app/generator")
generator_model = GeneratorModel(init=True)
sys.path.remove("./app/generator")


from app.generator_restful import generator_view