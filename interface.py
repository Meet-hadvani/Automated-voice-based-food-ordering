import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import pickle
import json
import speech_recognition as sr

def recordAudio():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		r.adjust_for_ambient_noise(source, duration=1)
		print("i'm listening ")
		audio = r.listen(source)
	data = ""
	try:
		data = r.recognize_google(audio)
	except sr.UnknownValueError:
		print("google speech recognition could not understand ")
	except sr.RequestError as e:
		print("Could not request results from google speech recognizer: {0}".format(e))
	return data

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1        
	return numpy.array(bag)

def order():
	print("\n\nStart ordering with pizza name and size\n")
	sizes = ["small", "medium", "large"]
	quantity = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
	cart = []
	while True:
		#inp = recordAudio()
		inp = input("you : ")
		split = inp.split(" ")
		size = [x for x in sizes if x in split]
		number_pizza = [x for x in quantity if x in split]
		print("you - ", inp)
        
		results = model.predict([bag_of_words(inp, words)])
		results_index = numpy.argmax(results)
		tag = labels[results_index]
		
		for tg in data["intents"]:
			if tg['tag'] == tag:
				cart.append([number_pizza, size, tag])
				print("cart - ", cart)
					
if __name__=="__main__":
	with open('intents.json') as file:
		data = json.load(file)
	with open('data.pickle','rb')as f:
		words, labels, training, output = pickle.load(f)

	tensorflow.reset_default_graph()
	net = tflearn.input_data(shape=[None, len(training[0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
	net = tflearn.regression(net)
	model = tflearn.DNN(net)
	print("menu")
	print("--------------------")
	print(" margarita pizza \n double cheese margarita pizza \n onion pizza \n garder delight pizza \n spring fling pizza \n lovers bite pizza \n sweet heat pizza")
	try:
		model.load("model.tflearn")
		order()
	except:
		print('error')