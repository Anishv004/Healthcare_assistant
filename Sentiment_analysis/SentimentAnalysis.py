from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
recognizer=sr.Recognizer()
with sr.Microphone() as source:
    print('Clearing background noise...')
    recognizer.adjust_for_ambient_noise(source,duration=1)
    print('Waiting for your message...')
    recordedaudio=recognizer.listen(source)
print('Done recording..')
print('Printing the message..')
text=recognizer.recognize_google(recordedaudio,language='en-US')
#print(text)
print('Your message: {}'.format(text))
#Sentiment analysis
Sentence=[str(text)]
analyser=SentimentIntensityAnalyzer()
for i in Sentence:
    v=analyser.polarity_scores(i)
for i in v:
    if i=='compound':
        val=v[i]
        if val>=-1 and val<-0.5:
            print("Negative Sentiment")
        elif val>=-0.5 and val<0.5:
            print("Neutral Sentiment")
        else:
            print("Positive Sentiment")
