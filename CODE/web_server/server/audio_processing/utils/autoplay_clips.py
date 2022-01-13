import vlc #python-vlc
import time

boat_data = []
critter = []
filepath_to_tarball = "C:/Users/gorge/Desktop/Fall 21/processed_data.tar/processed_data"
for i in range(10000,10100):
#for i in range(1900, 1950):
    sound_file = vlc.MediaPlayer(f"{filepath_to_tarball}/{i}/chip.wav")

    print(i)
    flag = False
    while not flag:
        sound_file.play()
        time.sleep(7)
        if input("was this nothing? (y/n)") == "y":
            flag=True
        elif input("Was this a boat(y/n)")=="y":
            boat_data.append(i)
            print(boat_data)
            flag=True
        elif input("Was this a animal(y/n)")=="y":
            critter.append(i)
            print(critter)
            flag=True
            
critterfile = open("critters.txt", "w")
for crit in critter:
    critterfile.write(crit + "\n")
critterfile.close()

boat_datafile = open("boatfile.txt", "w")
for boat in boat_data:
    boat_datafile.write(boat + "\n")
boat_datafile.close()