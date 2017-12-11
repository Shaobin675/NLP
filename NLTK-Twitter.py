#NLTK-Twitter.py
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

#consumer key, consumer secret, access token, access secret.
ckey = "PLEASE USE YOUR OWN TWITTER API"
csecret = "PLEASE USE YOUR OWN TWITTER API"
atoken = "PLEASE USE YOUR OWN TWITTER API"
asecret = "PLEASE USE YOUR OWN TWITTER API"

style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


class listener(StreamListener):

	def on_data(self, data):

		#print(data)
		all_data = json.loads(data)
		tweet = all_data['text']
		location = all_data['user']['location']
		print(location)
		if location == None:
			location = 'none'
		sentiment_value, confidence = s.sentiment(tweet)
		writein = location + ':::' + tweet + ':::' + sentiment_value + ':::' + str(confidence)
		print(writein)
		if confidence * 100 >= 80:
			output = open('twitter-car-out.txt', 'a')
			output.write(writein)
			output.write('\n')
			output.close()

		return True

	def on_error(self, status):
		print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["watermelon"])
