# Converts the videos to mp3 
import os 
import subprocess

files = os.listdir("videos") 
for file in files:
    print(file) 
    name = file.replace(".mkv", "")
    tutorial_number = name.split("_")[1]
    title = name.split("_", 2)[2].strip().lstrip("-").strip()
    output_name = f"{tutorial_number}_{title.replace(' ', '_')}.mp3"

    subprocess.run([
        "ffmpeg",
        "-i", f"videos/{file}",
        "-vn",
        f"audios/{output_name}"
    ])
    #tutorial_number = file.split(" [")[0].split(" #")[1]
    #file_name = file.split(" ｜ ")[0]
    #print( tutorial_number,  file_name)
    #subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])