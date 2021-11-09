import numpy as np 
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from sklearn.metrics.pairwise import cosine_similarity

 
'''dataset preparation
import numpy as np
import nltk
import re
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
def embed(input):
  return np.array(model([input]))
dataset['Question_Vector'] = dataset.Question.map(embed)
pickle.dump(dataset, open('dataset.pkl', 'wb'))
'''   
        
class DialogueManager(object):
    def __init__(self):
 
        #self.model = tf.saved_model.load("../data/tmp/mobilenet/1/")
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.dataset = pickle.load(open('dataset.pkl', mode='rb'))
         
        self.QUESTION_VECTORS =  self.dataset.Question_Vector 
        self.COSINE_THRESHOLD = 0.5
        
        self.chitchat_bot = ChatBot("Chatterbot")       
        trainer = ChatterBotCorpusTrainer(self.chitchat_bot)
        trainer.train("chatterbot.corpus.english")
 
         
        
    def embed(self,input):
        return np.array(self.model([input]))         
    
        
        
    def semantic_search(self, query):
        """Returns max_cos_ind and cosine similairty value
        query_vec = self.embed(query)
        sims = []
        for que_vec  in self.QUESTION_VECTORS:            
            sim =  cosine_similarity(query_vec, query_vec)
            sims.append(sim)
        max_ind = sims.index(max(sims))
        return max_ind,sims[max_ind][0][0]      
            
    
    def generate_answer(self, question):
        '''This will return list of all questions according to their similarity,but we'll pick topmost/most relevant question'''
        ind, most_relevant_sim_score = self.semantic_search(question)
         
        if most_relevant_sim_score >= self.COSINE_THRESHOLD:
            answer = self.dataset.Answer[ind]
        else:
            answer = self.chitchat_bot.get_response(question)
        return answer
      
         
