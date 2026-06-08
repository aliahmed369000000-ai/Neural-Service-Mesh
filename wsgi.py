from main import NeuralServiceMesh
from api.app import create_app

mesh = NeuralServiceMesh()
app = create_app(mesh)
