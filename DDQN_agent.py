import numpy as np
import cv2
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import os
tf.compat.v1.disable_eager_execution()

from collections import deque

GPUs = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUs:
    tf.config.experimental.set_memory_growth(gpu, True)


MAX_BATCH_MEMORY = 10
class Agent:
    def __init__(self):
        self.height = 96
        self.width = 96
        self.channels = 1
        self.action_space = [(-1,1,0.5),(0,1,0.5),(1,1,0.5),
                             (-1,1,0),(0,1,0),(1,1,0),
                             (-1,0,0.5),(0,0,0.5),(1,0,0.5),
                             (-1,0,0),(0,0,0),(1,0,0),]  #steer, gas, breaking
        self.gamma = 0.99
        self.learning_rate = 0.0003
        self.epsilon = 1.0
        self.epsilon_decrease = 0.9999
        self.epsilon_min = 0.05
        self.experience_replay_buffer = deque(maxlen=10000)
        self.model = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(self.height, self.width, self.channels)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))   #参数调整
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())

        # Dense layers
        self.model.add(Dense(units=216, activation='relu'))
        self.model.add(Dense(units=12, activation='linear'))

        # Compile the model with Huber loss
        self.model.compile(optimizer=Adam(learning_rate = self.learning_rate), loss=Huber())
        self.target_model = tf.keras.models.clone_model(self.model)

    def choose_action(self,state,best=False):     #没有加best
        state = np.expand_dims(state,axis = 0)
        action_index = np.argmax(self.model.predict(state)[0])

        if best: return self.action_space[action_index]

        sample = np.random.choice([0, 1], p=[1-self.epsilon, self.epsilon])
        if sample == 1:   #如果概率为epsilon就随机选择行动否则使用最佳行动
            action_index = random.randrange(len(self.action_space))   
        return self.action_space[action_index]
    
    def get_batch(self):
        best_batch = []
        number_list = list(range(1,len(self.experience_replay_buffer)+1))
        for _ in range(MAX_BATCH_MEMORY):
            total_number = len(number_list)*(len(number_list)+1)//2
            probs = []
            for i in range(len(number_list)):
                probs.append((i+1)/total_number)
            choice = np.random.choice(number_list,1,p = probs)[0]#就是按照概率梯度取前面的，排列
            del number_list[number_list.index(choice)]  #不要重复取
            best_batch.append(self.experience_replay_buffer[choice-1])
        return best_batch
            
    def experience(self):
        if len(self.experience_replay_buffer)>MAX_BATCH_MEMORY:
            batch_before = self.get_batch()
            
            train_state = []
            train_target = []
            for state, action, reward, next_state, done in batch_before:
                target = self.model.predict(np.expand_dims(state,axis = 0))[0]
                if done:
                    target[self.action_space.index(action)] = reward
                else:
                    ############ Double Deep Q Learning Here! #############
                    next_target = self.model.predict(np.expand_dims(next_state,axis = 0))[0]
                    target_pred = self.target_model.predict(np.expand_dims(next_state,axis = 0))[0]
                    target_index = np.where(next_target == np.max(next_target))[0][0] #取next_target中最大的那个的序号
                    target[self.action_space.index(action)] = reward + self.gamma * target_pred[ target_index ]
                train_state.append(state)
                train_target.append(target)
            # batch fitting
            self.model.fit( np.array(train_state), np.array(train_target), epochs=1, verbose=0)
            #减小探索欲望
            if self.epsilon>self.epsilon_min:
                self.epsilon *= self.epsilon_decrease

    def update_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def store_transition(self, state, action, reward, next_state, done):
        self.experience_replay_buffer.append((state, action, reward, next_state, done))
    
    def save_model(self,count,data):
        model_direction = "./model_self"
        data_direction = "./reward_self"
        if not os.path.exists( model_direction ):
             os.makedirs( model_direction )
        self.target_model.save_weights( model_direction + f"/episode_{count}" + ".h5" )

        # saving results
        if not os.path.exists( data_direction ):
             os.makedirs( data_direction )
        np.savetxt(f"{data_direction}" + f"/episode_{count}" + ".csv", data, delimiter=",")
    
    def load_model(self, name):
        self.model.load_weights(name)
        self.model.set_weights( self.model.get_weights() )


def convert_to_grey(state):
    x, y, _ = state.shape
    after_cut = state[0:int(y*0.85), 0:x] #将下面的黑条去掉
    lower_grey = np.array([100, 100, 100])#灰度下界
    upper_grey = np.array([150, 150, 150])#灰度上界
    mask = cv2.inRange(after_cut, lower_grey, upper_grey)
    # cv2.imwrite("RGB_image.jpg", after_cut)
    # cv2.imwrite("gray_image.jpg", mask)
    # os.system("pause")
    grey = cv2.cvtColor(state,  cv2.COLOR_BGR2GRAY)
    grey = grey.astype(float)
    grey_normalised = grey/ 255


    return [np.expand_dims( grey_normalised, axis=2 ), np.any(mask== 255)]


