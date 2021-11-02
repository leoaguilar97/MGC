from pydub import AudioSegment
import os

cwd = os.getcwd()

print("Iniciando proceso de construccion de DB")
genres = ["blues", "classical", "country", "jazz", "rock"]

input_format = "./db/{genre}/{genre}.{file_number}.au"
output_format = "./db/{genre}/{genre}.{file_number}.mp3"

i = 0
for genre in genres:
    for i in range(100):
        file_number = (str(i)).zfill(5)

        file_name = input_format.format(genre=genre, file_number=file_number)
        output_name = output_format.format(genre=genre, file_number=file_number)

        print(f"{file_name} > {output_name}")
        au_audio = AudioSegment.from_file(file_name, format="au")
        au_audio.export(output_name, format="mp3")
