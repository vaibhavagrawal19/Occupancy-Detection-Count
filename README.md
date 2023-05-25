# Occupancy-Detection-Count

This is the server-side code of a project which aims to detect the occupany (the number of people present) in a room using a very low resolution thermal camera. We have used a Convolutional Neural Network, and the trained network is given.

The expected input is 8x8 temperature values in Celsius. The intended output is the number of people predicted.

## How does this work?
We use cubic interpolation of the 8x8 temperature values followed by temperature-color mapping to generate a 32x32 image. Then we use a Convolutional Neural Network to predict the number of people present inside the room. The CNN has been trained on numerous training examples. 

For anyone interested, please contact me via the email.
