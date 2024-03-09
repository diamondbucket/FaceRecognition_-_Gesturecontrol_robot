from serial import Serial
arduinoData = Serial('com3',115200)

while True:
    cmd = input('please enter your command:')
    cmd=cmd+'\r'
    arduinoData.write(cmd.encode())