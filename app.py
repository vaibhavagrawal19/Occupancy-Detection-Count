from flask import Flask, render_template, request, redirect
import functions
from PIL import Image
import shutil
import tensorflow as tf
max_values = [24.25] * 64
thermal_cam = [24.25] * 64
model = tf.keras.models.load_model("static/trained_cnn3")
count = [0]
app = Flask(__name__)

database = [{"room_no": "117", "password": "lite"}]


@app.route("/calibrate")
def calibrate():
    base = count[0]
    for i in range(64):
        max_values[i] = 0
    while count[0] - base < 5:
        for i in range(64):
            max_values[i] = max(max_values[i], thermal_cam[i])
    print("calibrated successfully!!")
    return "200"


@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/index")
def home_page():
    return render_template("index.html")


@app.route("/form_login", methods=["GET", "POST"])
def login_procedure():
    username = request.form.get("room_no")
    password = request.form.get("password")
    print(username)
    print(password)
    found = False
    for i in database:
        if i["room_no"] == username and i["password"] == password:
            return redirect("/index")
        else:
            render_template("incorrect.html")


@app.route("/register_page")
def register():
    return render_template("register.html")


@app.route("/form_register", methods=["GET", "POST"])
def register_procedure():
    username = request.form.get("room_no")
    password = request.form.get("password")
    database.append({"room_no": username, "password": password})
    return redirect("/index")


@app.route("/obtain", methods=["GET", "POST"])
def obtain_count():
    count[0] = count[0] + 1
    for i in range(10):
        thermal_cam[i] = float(request.form.get("val0" + str(i)))
    for i in range(10, 64):
        thermal_cam[i] = float(request.form.get("val" + str(i)))
    functions.generate_img(thermal_cam, max_values)
    value = functions.predict_count(model)
    shutil.copy("static/" + str(value) + ".png", "static/curr0.png")
    shutil.copy("static/" + str(value) + ".png", "static/curr1.png")
    functions.post_to_om2m()
    return "200"


@app.route("/status", methods=['GET', 'POST'])
def index():
    return render_template("data.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/contact")
def contact_page():
    return render_template("contact.html")


@app.route("/motivation")
def motivation_page():
    return render_template("motivation.html")


@app.route("/team")
def team_page():
    return render_template("team.html")

app.run(host="192.168.252.199")
