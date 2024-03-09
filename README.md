for classification visit "https://github.com/Mjrovai/OpenCV-Face-Recognition" as i saw his tutorial and wrote this code..
and i didnt use an esp8266 because i didnt have it so instead i used an arduino uno
for my convinience i created a venv for python as i wasnt able to install media pipe for my default python version that was 3.12
A folder dataset has to be created inorder to store the datasets derived from the dataset.py
for higher acuracy change the "elif statement's count to more that 30 this will ensure the model is trained properly for accurate facial recognition

I used Serial for connecting my arduino to my python script as  this is my first prohect i just used a single variable as a bus to send data and triggered some function in arduino when the data from python script is passed
thus helping me with controlling the arduino using a python script

initially this was a line follower hence the Analog pins.. i didnt want to remove them hence i used them as decorations or visual indicator for the bot's movement.
