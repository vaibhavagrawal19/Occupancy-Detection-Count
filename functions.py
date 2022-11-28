import requests
import os
import time
from scipy.interpolate import griddata
import math
import numpy as np
from colour import Color
from PIL import Image
import shutil
from flask import request
from onem2m import *
import json
import matplotlib as plt

last_prediction = 0

cse_ip = "esw-onem2m.iiit.ac.in"
cse_port = 443
# om2m_origin = "ZwxGF8:czI1nQ"
om2m_origin = "ZwxGF8:czI1nQ"
om2m_mn = "/~/in-cse/in-name/"
om2m_ae = "Team-28"
om2m_data_cont = "occupants-0"
server = "https://" + cse_ip + ":" + str(cse_port) + om2m_mn


def create_cnt():
    headers = {
        "X-M2M-Origin": om2m_origin,
        "Content-Type": "application/json;ty=3"
    }
    payload = {
        "m2m:cnt": {
            "rn": "trial-1",
            "mni": 10000,
        }
    }
    try:
        print(server + "Team-28")
        response = requests.post(server + "Team-28", json=payload, headers=headers)
    except TypeError:
        response = requests.post(server + "Team-28", json=payload.dump(), headers=headers)
    print(str(response.status_code))
    print(str(response.text))
    return


def post_to_om2m(mydata):
    headers = {
        "X-M2M-Origin": om2m_origin,
        "Content-Type": "application/json;ty=4"
    }
    data = str(last_prediction)
    payload = {
        "m2m:cin": {
            "con": str(mydata),
            "lbl": "V1.0.0",
            "cnf": "text"
        }
    }
    response = requests.post(server + "Team-28/trial-1",
                  json=payload, headers=headers)
    if (str(response.code).startswith("20")):
        print("POST OK")
    return

values = list()
timestamps = list()


def get_from_onem2m():
    headers = {
        "X-M2M-Origin": om2m_origin,
        "Content-Type": "application/json;ty=4"
    }
    response = requests.get(server + "Team-28/trial-1/?rcn=4", headers=headers)
    print(str(response.status_code))
    if (str(response.status_code).startswith("20")):
        print("GET OK")
    # print(str(response.text))
    m2mcin = response.json()["m2m:cnt"]["m2m:cin"]
    values = list()
    timestamps = list()
    for i, j in enumerate(m2mcin):
        if i % 2 == 1:
            timestamps.append(j["con"])
        else:
            values.append(j["con"])


def plot_and_capture():
    xpoints = timestamps
    ypoints = values
    plt.plot(xpoints, ypoints)
    plt.savefig("plot.jpg")


# constrain the values of the colours within the limits
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

# map the value of the temperature to the colour
def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def generate_img(thermal_cam, max_values):
    COLORDEPTH = 1024

    # the points in the 8x8 grid
    points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]

    # the grid to store the 240x240 pixel values
    grid_x, grid_y = np.mgrid[0:7:240j, 0:7:240j]

    # listing the colours
    blue = Color("indigo")
    # the list of all colours from blue (indigo) to red with the depth as COLORDEPTH
    colors = list(blue.range_to(Color("red"), COLORDEPTH))
    # splitting into R, G, B
    colors = [(int(c.red * 255), int(c.green * 255), int(c.blue) * 255)
              for c in colors]
    for i in range(64):
        thermal_cam[i] = float(thermal_cam[i])
        MINTEMP = max_values[i]
        row = i / 8
        col = i % 8
        if row >= 1 and row <= 6 and col >= 1 and col <= 6:
            thermal_cam[i] = map_value(
                thermal_cam[i], MINTEMP, MINTEMP + 1.0, 0, COLORDEPTH - 1)
        else:
            thermal_cam[i] = map_value(
                thermal_cam[i], MINTEMP, MINTEMP + 0.5, 0, COLORDEPTH - 1)
    bicubic = griddata(points, thermal_cam, (grid_x, grid_y), method="cubic")
    image_arr = np.zeros((240, 240, 3), dtype=np.uint8)
    for ix, row in enumerate(bicubic):
        for jx, pixel in enumerate(row):
            image_arr[ix][jx] = colors[constrain(
                int(pixel), 0, COLORDEPTH - 1)]
    image_render = Image.fromarray(image_arr)
    image_render.save("static/thermal_img0.jpg")
    image_render.save("static/thermal_img1.jpg")
    image_render = image_render.resize((60, 60))
    image_render.save("static/thermal_img.jpg")
    return "200"


def predict_count(model):
    import tensorflow as tf
    from tensorflow import keras
    img = tf.keras.utils.load_img(
        'static/thermal_img.jpg', target_size=(240, 240))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['0', '1', '2', '3', '4']
    answer = class_names[np.argmax(score)], 100 * np.max(score)
    answer = list(answer)
    answer = answer[0]
    last_prediction = answer
    with open("static/prediction.txt", "w") as f:
        f.write(str(answer))
    return answer
