from model_files import temp
from flask import Flask, request, render_template, send_from_directory
from markupsafe import escape

#flask --app server/server.py run
temp.foo_bar(12)

app = Flask(__name__, static_folder='../client/src')

@app.route("/") #OR can do app.get
def serve():
    return send_from_directory(app.static_folder, "index.tsx") #Gotta figure out how to load html/jsx files for frontend

@app.route("/<name>")
def hello(name):
    return f"Hello, {escape(name)}!" #Prevent script injecting

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return "do_the_login()"
    else:
        return "show_the_login_form"
    

# if __name__ == "__main__":
#     app.run()