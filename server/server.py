from flask import Flask, request, render_template, send_from_directory
from markupsafe import escape
from get_review_score_and_prediction import get_review_score_pred
from os import path

#flask --app server/server.py run
#use python3 server.py

#Consts/setup vars are stored here
#TODO: probably moving this to separate file, CONSTANTS.
MODEL_NAME = "dummy.pth"
test_file_path = 'model_files/data/test-00000-of-00001.parquet'

# We create this on setup before anything else
#TODO: update the split to have all in on one dl. Fix !!! issue in get first then update this
# dataLoaders_and_vocab = setup_data.create_dataLoaders(test_file_path, 1, 0.25, 0.00020, True)
# test_dl = dataLoaders_and_vocab[1]

app = Flask(__name__, 
            static_url_path='',
            static_folder='../client/build',
            template_folder='build')

@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def serve(path):
    if path != "" and path.exists(path.join(app.static_folder, path)):
        print("je")
        return send_from_directory(app.static_folder, path)
    else:
        print("jdsfgsdgdsfge")
        return send_from_directory(app.static_folder, 'index.html')


#TEsting server stuff
@app.route("/<name>")
def hello(name):
    return f"Hello, {escape(name)}!" #Prevent script injecting

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         return "do_the_login()"
#     else:
#         return "show_the_login_form"

# So do we add in methods? Probably. That would be a good thing. 

@app.route('/data')
def get_data():
    result = get_review_score_pred(MODEL_NAME)
    return result

if __name__ == "__main__":
    app.run(debug=True)