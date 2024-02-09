import speech_recognition as sr

r = sr.Recognizer() 
f = open('C:/Users/kssan/OneDrive/Desktop/Final_proj_scrape/WORKING_STUFF/demo.txt','w+')
with sr.Microphone() as source:
    print('Speak Anything : ')
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print('You said: {}'.format(text))
        f.write(format(text))
        
    except:
        print('Sorry could not hear')

f.close()