#main
#working
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download('punkt')

chatbot=ChatBot('Chatbot')
trainer = ChatterBotCorpusTrainer(chatbot)

def nltk_test(inp):
  tokenized_word=word_tokenize(inp)
  stopwrds=list(stopwords.words("english"))
  extra=['I','Can']
  stop_words=stopwrds+extra
  filtered_sent=[]
  for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
  x=" ".join(filtered_sent)
  return x

trainer.train(
    # "./convo.yml"
    # "chatterbot.corpus.english",
    "chatterbot.corpus.english.greetings",
    "chatterbot.corpus.english.conversations"
)

trainer = ListTrainer(chatbot)


data=['What should I do if someone is choking?',
      "If someone is choking and can't breathe or talk, you should perform the Heimlich maneuver.",
      "How do I treat a burn?",
      "You should run cool water over the burn for at least 10 minutes and cover it with a sterile bandage. Do not apply ice or butter to the burn.",
      "What should I do if someone has a seizure?",
      "If someone has a seizure, you should stay with them and make sure they are safe. Do not restrain them or put anything in their mouth. Call for medical help if the seizure lasts more than 5 minutes or if it is their first seizure.",
      "I have a headache.",
      "Do you have any other symptoms?",
      "Yes, I also have a fever and chills.",
      "It sounds like you might have the flu. Please see a doctor for diagnosis and treatment.",
      "My stomach hurts.",
      "Is it a sharp pain or a dull ache?",
      "It's a sharp pain.",
      "You might have an ulcer or gastritis. Please see a doctor for diagnosis and treatment.",
      "I feel dizzy and lightheaded.",
      "You might have low blood pressure. Please see a doctor for diagnosis and treatment.",
      "What are the side effects of aspirin?",
      "The side effects of aspirin include stomach upset, heartburn, and drowsiness.",
      "How do I take this medication?",
      "You should take this medication with food or milk to reduce stomach upset. Follow the dosage instructions on the label or as prescribed by your doctor.",
      "Can I drink alcohol while taking this medication?",
      "It depends on the medication. Some medications can have dangerous interactions with alcohol. Check with your doctor or pharmacist for advice."
      ]


# data=data0+data1+data2+data3

data_prime=[]

for i in range(0,len(data),2):
  temp=nltk_test(data[i])
  data_prime.append(temp)
for i in range(1,2*len(data_prime)):
  if i%2!=0:
      data_prime.insert(i,data[i])
print(data_prime) 


trainer.train(data)
trainer.train(data_prime)


while True:
  inp=input("User :")
  response = chatbot.get_response(nltk_test(inp.lower()))
  print("Bot : ", str(response))