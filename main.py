import speech_recognition as sr
import pyttsx3
import jsonDataPull

def speak(text):
 engine.say(text)
 engine.runAndWait()


r = sr.Recognizer()


engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
engine.setProperty("rate", 120)

while(1):
 with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=0.2)
    print("Talk")
    audio_text = r.listen(source)
    print("Time Over")
    try:
     speechText = r.recognize_google(audio_text)
     print("Text: " + speechText)
     speak(jsonDataPull.nlpDrama(speechText))
    except:
     continue
