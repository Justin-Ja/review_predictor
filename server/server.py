from flask import Flask, request, render_template, send_from_directory
from markupsafe import escape

#flask --app server/server.py run
#use python3 server.py

app = Flask(__name__, 
            static_url_path='',
            static_folder='../client/build',
            template_folder='build')

@app.route("/") #OR can do app.get
def serve():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<name>")
def hello(name):
    return f"Hello, {escape(name)}!" #Prevent script injecting

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return "do_the_login()"
    else:
        return "show_the_login_form"
    
@app.route('/data')
def get_example():
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":"x", 
        "programming":"python"
    }

if __name__ == "__main__":
    app.run(debug=True)