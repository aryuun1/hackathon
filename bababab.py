import whisper

model = whisper.load_model("base")
result = model.transcribe("sample.mp4")
print(result["text"])