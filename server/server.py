from flask import Flask, request, render_template, send_from_directory
from markupsafe import escape
from get_review_score_and_prediction import get_review_score_pred
import os
from CONSTANTS import MODEL_NAME, MODEL_PATH
#flask --app server/server.py run
#use python3 server.py

# We create this on setup before anything else
app = Flask(__name__, 
            static_url_path='',
            static_folder='../client/build',
            template_folder='build')

@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         return "do_the_login()"
#     else:
#         return "show_the_login_form"

# So do we add in methods? Probably. That would be a good thing. 

@app.route('/data')
def get_data():
    result = get_review_score_pred(MODEL_NAME, MODEL_PATH)
    return result

if __name__ == "__main__":
    app.run(debug=True)