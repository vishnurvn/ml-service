import base64
from io import BytesIO

import torch
from PIL import Image
from flask import Flask, render_template, url_for, request, redirect, session
from torchvision.transforms import ToTensor

from file_ops import Net

app = Flask(__name__)
net = Net()
net.load_state_dict(torch.load("./model.h5"))
to_tensor = ToTensor()
app.config["SECRET_KEY"] = "my_secret_key"


@app.route("/")
def home():
    result = None
    if "result" in session:
        result = {
            "class": session["result"],
            "image_string": session["image_string"]
        }
        del session["result"]
        del session["image_string"]
    return render_template("main.html", result=result)


@app.route("/upload", methods=["POST"])
def upload_image():
    data = request.files["image"]
    bytes_io = BytesIO(data.stream.read())
    image = Image.open(bytes_io)
    image_tensor = to_tensor(image)
    assert image_tensor.size() == (1, 28, 28)
    image_tensor = image_tensor.reshape((1, 1, 28, 28))
    output = net(image_tensor)
    session["result"] = output.argmax().item()
    session["image_string"] = base64.b64encode(bytes_io.getvalue()).decode('utf-8')
    return redirect(url_for('home'))


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html"), 500


if __name__ == '__main__':
    app.run()
