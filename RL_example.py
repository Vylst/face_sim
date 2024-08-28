import keras
import gym
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import InputLayer, Dense

env = gym.make('NChain-v0')

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 5)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# now execute the q learning
num_episodes = 100
y = 0.95
eps = 0.5
decay_factor = 0.999
r_avg_list = []

for i in range(num_episodes):
	
	#Reset environment
	s = env.reset()
	
	#
	eps *= decay_factor
	
	if i % 100 == 0:
		print("Episode {} of {}".format(i + 1, num_episodes))
	done = False
	r_sum = 0
	
	while not done:
	
		#Choose action (a), either it will be a random value or be chosen based on the prediction of 
		if np.random.random() < eps:
			a = np.random.randint(0, 2)
		else:
			#Predict the Q values for each possible action. Choose the action corresponding to highest Q-value
			a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
		    
		
		#Do action a -> That makes agent obtain a reward (r) and also go to a new state (new_s). Done indicates end of game (number of episodes)
		new_s, r, done, _ = env.step(a)
		
		
		#Q-learn updating rule -> 
		target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
		target_vec = model.predict(np.identity(5)[s:s + 1])[0]
		target_vec[a] = target
		#print(np.shape(target))
		#print(target_vec.reshape(-1, 2))
		model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
		
		#Update state
		s = new_s
		#Update cumulative reward
		r_sum += r
	
		print('done:', done)
		
	#Save average reward per episode
	r_avg_list.append(r_sum / 1000)

plt.plot(r_avg_list)
plt.ylabel('Average reward per game')
plt.xlabel('Number of games')
plt.show()
for i in range(5):
	print("State {} - action {}".format(i, model.predict(np.identity(5)[i:i + 1])))
    
    
