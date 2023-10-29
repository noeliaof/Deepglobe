import os
from inference import *

from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#load model
config = load_config('/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/config.yaml')
model = build_model(config)
model_state_dict = torch.load('/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/models/final_model.pth', map_location='cpu')
model.load_state_dict(model_state_dict)
model.eval()


@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'samples')
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))

    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    print("Destination: ",destination)

    image_list = []
    image_list.append(destination)

    result =  inference_single_image(model, image_list[0])
    result = result.squeeze()

    file_outname = '%s_out.png'%(filename[:-4])
    
    return render_template("complete.html", image_name=file_outname)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5001"))



