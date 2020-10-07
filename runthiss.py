# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:29:35 2019
用来实现布洛赫球面上的归零操作 归到 |1>态
此程序可以生成在特定的起点所施加的操作
生成在操作下量子态及其在布洛赫球上位置的变化
@author: Waikikilick
"""

from environment_noise import Env
from Net_dql import DeepQNetwork
import warnings
warnings.filterwarnings('ignore')
from time import *

import numpy as np
np.random.seed(1)


#在布洛赫球上选取一系列训练点
#在 env 文件中会看到这些点是随角度均匀分布的
theta_num = 5  #theta 选取点的数目
#将范围在 [0，pi] 之间的 theta 包含起止点均匀选取 theta_num 个点
varpsi_num = 10 #varpsi 选取点的数目
#将范围在 [0，2*pi) 之间的 varpsi 包含起点但不包含终点均匀选取 varpsi_num 个点
fidelity_list = np.zeros((theta_num, varpsi_num)) 
#记录每个选择测试点在经过操作之后可以达到的最终保真度


#在布洛赫球面上选取一系列测试点
#选取方法是在上面均匀分布的训练点的间隔中，等间距选择 _mul 个点
#训练集和测试集是不交叉的
test_theta_mul = 2
test_varpsi_mul = 4

test_theta_num = test_theta_mul*(theta_num-1) #测试集 test_theta 选取点的数目
test_varpsi_num = varpsi_num*test_varpsi_mul  #      test_varpsi 选取点的数目
test_fidelity_list = np.zeros((test_theta_num, test_varpsi_num))
# #记录每个选择测试点在经过操作之后可以达到的最终保真度


#--------------------------------------------------------------------------------------
#训练部分
def training():
    
    
    print('\ntraining...')
    
    #根据之前选好的初始点，依次训练神经网络，得到对应的最大最终保真度，并将其记录在矩阵中
    for k in range(theta_num):#按之前生成的 theta 数，依次训练
        for kk in range(varpsi_num):#.....varpsi..........
            print(k,kk)
            env.rrset(k,kk) #在训练完一个选好的训练点后，将初量子态调到下一个训练点上
            global tanxin
            tanxin = 0 #动作选择的 epsilon:当训练时，选择一个训练点后，tanxin = 0 epsilon 设为 0，之后改变 tanxin = 0.5 施加递减的贪心策略；
                        #更换训练点后，tanxin = 0，epsilon 重新选为0，重新执行递减贪心策略
                        #测试时，tanxin = 1，epsilon = 1,不使用贪心策略 ，直接选择值最大的动作
            
            ep_max = 100
            for episode in range(ep_max):
                observation = env.reset()

                while True: 
                    
                    action = RL.choose_action(observation,tanxin)
                    observation_, reward, done, fid = env.step(action)
                    RL.store_transition(observation, action, reward, observation_)
                    RL.learn()
                    tanxin = 0.5

                    observation = observation_
                    if done:
                        
                        break  
                    
#----------------------------------------------------------------------------------
    
#---------------------------------------------------------------------------------- 
#测试部分(无噪声)
    
def testing():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting...\n')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到对应的最大最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            test_fid = 0
            
            while True:
                action = RL.choose_action(observation,tanxin)
                observation_, reward, done, fid = env.step(action)
                observation = observation_
                test_fid = max(test_fid,fid) 
                #直接选择在操作过程中最佳保真度作为本回合的保真度
                #因为在计算过程中，达到最佳保真度这一点是可以保证做到的
                
                if done:
                    break
                        
            test_fidelity_list[k,kk] = test_fid #将最大的最终保真度记录到矩阵中
    return test_fidelity_list
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
#测试部分（ J 噪声环境）
    
def testing_noise_J():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting...noise_J\n')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到对应的噪声环境下的最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            action_list = [] #用来保存本回合所采取的动作，用于噪声分析
            fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
            while True:
                action = RL.choose_action(observation,tanxin)
                # action = 0 #可以测试没有噪声时，是否和预期一致，检验算法正误
                action_list = np.append(action_list,action)
                observation_, reward, done, fid = env.step(action)
                fid_list = np.append(fid_list,fid)
                observation = observation_
                #这个环节就是完全根据网络的预测来选择动作，并将动作和对应的保真度记录下来
                if done:
                    break
                
            #下面根据记录的保真度，挑出在哪一步保真度最高来进行截取动作（因为有可能达到最高保真度之后，智能体又多走
            #将噪声加到达到最佳保真度之前各步数上
            observation = env.reset()
            fid_list = list(map(float,fid_list))
            max_index = fid_list.index(max(fid_list))
            action_list = action_list[0:max_index+1]
            
            
            #加入噪声
            test_fid_noise = 0
            for action in action_list:
                
                observation_, reward, done, fid = env.step_noise_J(action)
                observation = observation_
                test_fid_noise = fid #选择最后一步的保真度作为本回合的保真度
                
            test_fidelity_list[k,kk] = test_fid_noise #将最终保真度记录到矩阵中
    return test_fidelity_list
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#测试部分（ h 静态噪声环境）
    
def testing_noise_h_s():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting noise_h_s...')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到在环境噪声下的最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            action_list = [] #用来保存本回合所采取的动作，用于噪声分析
            fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
            while True:
                action = RL.choose_action(observation,tanxin)
                
                action_list = np.append(action_list,action)
                observation_, reward, done, fid = env.step(action)
                fid_list = np.append(fid_list,fid)
                observation = observation_
                
                if done:
                    break
                
            observation = env.reset()
            fid_list = list(map(float,fid_list))
            max_index = fid_list.index(max(fid_list))
            action_list = action_list[0:max_index+1]
            
            
            test_fid_noise = 0
            for action in action_list:
                
                observation_, reward, done, fid = env.step_noise_h_s(action)
                observation = observation_
                test_fid_noise = fid 
                
            test_fidelity_list[k,kk] = test_fid_noise 
    return test_fidelity_list
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#测试部分（ J 静态噪声环境）
    
def testing_noise_J_s():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting noise_h_s...')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到在环境噪声下的最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            action_list = [] #用来保存本回合所采取的动作，用于噪声分析
            fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
            while True:
                action = RL.choose_action(observation,tanxin)
                
                action_list = np.append(action_list,action)
                observation_, reward, done, fid = env.step(action)
                fid_list = np.append(fid_list,fid)
                observation = observation_
                
                if done:
                    break
                
            observation = env.reset()
            fid_list = list(map(float,fid_list))
            max_index = fid_list.index(max(fid_list))
            action_list = action_list[0:max_index+1]
            
            
            test_fid_noise = 0
            for action in action_list:
                
                observation_, reward, done, fid = env.step_noise_J_s(action)
                observation = observation_
                test_fid_noise = fid 
                
            test_fidelity_list[k,kk] = test_fid_noise 
    return test_fidelity_list
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#测试部分（ h 噪声环境）
    
def testing_noise_h():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting noise_h...')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到在环境噪声下的最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            action_list = [] #用来保存本回合所采取的动作，用于噪声分析
            fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
            while True:
                action = RL.choose_action(observation,tanxin)
                # action = 0
                action_list = np.append(action_list,action)
                observation_, reward, done, fid = env.step(action)
                fid_list = np.append(fid_list,fid)
                observation = observation_
                
                if done:
                    break
                
            observation = env.reset()
            fid_list = list(map(float,fid_list))
            max_index = fid_list.index(max(fid_list))
            action_list = action_list[0:max_index+1]
            
            
            test_fid_noise = 0
            for action in action_list:
                
                observation_, reward, done, fid = env.step_noise_h(action)
                observation = observation_
                test_fid_noise = fid 
                
            test_fidelity_list[k,kk] = test_fid_noise 
    return test_fidelity_list
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
#测试某一特定点，分为两个部分：

#1. 得出该点在布洛赫球上的坐标值、最终保真度，优化脉冲

def testing_point(test_point_theta_ord,test_point_varpsi_ord):
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    global tanxin
    tanxin = 1
    
    actions = np.zeros((1,50)) - 1 #一共最多需要40次操作，但为了保险，多记录一点也无妨，没施加操作的位置都设为了 -1 以便于识别。
    action_step = 0
    states = np.zeros((50,4))


    #根据之前选好待测试点，依次输入神经网络，得到对应的最大最终保真度，并将其记录在矩阵中
    env.trrset(test_point_theta_ord,test_point_varpsi_ord)#将量子态重置到待测点上
    test_point_theta, test_point_varpsi = \
        env.test_theta[test_point_theta_ord], env.test_varpsi[test_point_varpsi_ord]
        
    print('test_point_theta:\n',test_point_theta,'\n\ntest_point_varpsi:\n',test_point_varpsi,'\n')
    #打印出选出来测试点
    observation = env.reset()
    states[0,:] = observation
    test_point_fid = 0
    
    while True:
        action = RL.choose_action(observation,tanxin)
        actions[0][action_step] = action  
        action_step += 1
        observation_, reward, done, fid = env.step(action)
        observation = observation_
        states[action_step] = observation
        test_point_fid = max(test_point_fid,fid)
        if done:
            break
    return test_point_fid, actions, states

#---------------------------------------------------------------------------

#2.得出该测试点在操作下，量子态的变化以及在布洛赫球上的位置
#输入量子态 states, 输出量子态在布洛赫球上的位置

def positions(states):
    #输入 states 的矩阵，行数为位置数，共 4 列标志着量子态的向量表示 [[1+2j],[3+4j]] 表示为：[1,3,2,4]
    #所以先将 state 表示变为 psi 表示 
    # b 矩阵第一列为 alpha,第二列为 beta
    b = np.zeros((states.shape[0],2),complex) 
    b[:,0] = states[:,0] + states[:,2]*1j
    b[:,1] = states[:,1] + states[:,3]*1j
    alpha = b[:,0]
    beta = b[:,1]
    
    #根据 alpha 和 beta 求直角坐标系下量子态的坐标
    z = 2*(alpha*alpha.conj())-1 #后面表示 z 的列向量中有多余的 -1 量，就是这里的原因。
    x = (beta*np.sqrt(2*(z+1))+beta.conj()*np.sqrt(2*(z+1)))/2
    y = (beta*np.sqrt(2*(z+1))-beta.conj()*np.sqrt(2*(z+1)))/(2*1j)
    
    #positions 矩阵为 位置数*行，3列，分别为 x,y,z
    positions = np.zeros((states.shape[0],3))
    positions[:,0], positions[:,1],positions[:,2] = x,y,z
    return positions
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
#将测试集的保真度从小到大排列出来，来展示保真度分布
def sort_fid(test_fidelity_list):
    sort_fid = np.zeros((1,))
    for i in range (test_fidelity_list.shape[0]):
        b = test_fidelity_list[i,:]
        sort_fid  = np.append(sort_fid,b)
    sort_fid.sort()
    return sort_fid
#--------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
#主程序部分

if __name__ == "__main__":

    
    dt_=np.pi/40
    noise_a = 0 #噪声的幅值
    env = Env(action_space=list(range(4)),   #允许的动作数 0 ~ 4-1 也就是 4 个
               dt=dt_, theta_num=theta_num , varpsi_num=varpsi_num,
               test_theta_mul=test_theta_mul , test_varpsi_mul=test_varpsi_mul,
               noise_a=noise_a)              
        
    RL = DeepQNetwork(env.n_actions, env.n_features,
              learning_rate=0.0005,
              reward_decay=0.9, 
              e_greedy=0.99,
              replace_target_iter=250,
              memory_size=3000,
              e_greedy_increment=0.0001 , batch_size=32
              )
    begin_training = time()
    fidelity_list = training() #训练
    end_training = time()
    training_time = end_training - begin_training
    print('\ntraing_time =',training_time)
    # print("\nFinal_fidelity=\n", fidelity_list,'\n')
    
    
    begin_testing = time()
    test_fidelity_list = testing() #测试
    # test_fidelity_list = testing_noise_J() #测试 J 噪声的影响
    # test_fidelity_list = testing_noise_h() #测试 h 噪声的影响
    end_testing = time()
    testing_time = end_testing - begin_testing
    print('\ntesting_time =',testing_time)
    
    
    # print('\ntest_fidelity_list:\n',test_fidelity_list)
    print('\nmean_test_fidelity:\n',np.mean(test_fidelity_list))
    
    #----------------
    #输出单个测试点来得到对其的操作、量子态及在布洛赫球上位置的变化
    # test_point_fid, actions, states = testing_point(-2,-10) #此处（-2，-10）点的意思是挑选上面测试点列表中的对应点来操作
    # print('\ntest_point_fid:\n',test_point_fid,'\ntest_point_actions:\n',actions)
    # print('\npositions:\n',positions(states))
    
    #----------------
    # mean_test_fidelity:
    #     0.9537978370221788
    #将测试集的保真度从小到大排列出来
    # print(sort_fid(test_fidelity_list))
    
     # 0.59028084 
     
     # 0.61871031 
     
     # 0.65986186 
     
     # 0.72882734 
     
     # 0.77209868 
     
     # 0.77702214 
     
     # 0.80242256 
     
     # 0.82025211 
     
     # 0.82605242 0.83185308 
     
     # 0.8358844
     
     # 0.86329545 
     
     # 0.86800214 0.87210138 
     
     # 0.88585056 0.88814156 
     
     # 0.89743203 0.90177722 
     
     # 0.90526826 0.90569848 0.90648903 0.9068744  0.90901118 0.91414223
     # 0.91396866  
     
     # 0.91502262 0.91503156 0.91684347 0.91873498 0.91976374 0.92367413 
     # 0.92409959 
     
     # 0.92527637 0.92633619 0.92670852 0.92910633 0.93016164 0.9331038  
     # 0.93405792 
     
     # 0.93746814 0.93836176 0.94450715 
     
     # 0.94539657 0.94560123 0.94587309 0.94659824 
     
     # 0.95560522 0.95628955 0.95648941 0.95703879 0.95809141 0.95840406 
     # 0.96209985 0.96220523 0.96348217 0.96359532 0.96470372 0.96157101
     
     # 0.96629032
     # 0.96771327 0.96772373 0.96812905 0.96856408 0.97047525 0.97199487
     # 0.97228747 0.9729248  0.97326857 0.97363231 0.97386974 0.97497657
     
     # 0.9772411  0.9777551  0.97775858 0.97778007 0.97799987 0.9795541
     # 0.98131691 0.98164636 0.98164636 0.98205134 0.98206978 0.98232812
     # 0.98251083 0.98268529 0.98288901 0.98360319 0.98391349 0.98399798
     # 0.98406931 0.98417283 0.98425577 0.98446919 0.98472714 0.98473022
     # 0.98490053
     
     # 0.98528415 0.98574733 0.98586688 0.98591502 0.98650302 0.99497613
     # 0.98676951 0.987046   0.98744469 0.98759298 0.98761079 0.98822217
     # 0.98837781 0.98847557 0.98851286 0.98851286 0.98855371 0.9885579
     # 0.98856715 0.98867214 0.98869279 0.98875578 0.98891603 0.98907118
     # 0.98921235 0.98922679 0.98943775 0.98992476 0.99003125 0.99003791
     # 0.99016029 0.9901665  0.9902208  0.99023009 0.99024698 0.99034746
     # 0.99037266 0.99038319 0.99039488 0.99044022 0.99057361 0.99068693
     # 0.99070551 0.99078813 0.99080048 0.99084536 0.99087488 0.99087488
     # 0.99097185 0.99098435 0.99106958 0.99116003 0.99118016 0.9911879
     # 0.99131407 0.99136601 0.9913922  0.99142697 0.99149089 0.99162756
     # 0.99170724 0.99175249 0.99177793 0.99185694 0.99191129 0.99206207
     # 0.99215193 0.99216823 0.99222647 0.9922372  0.99243768 0.99247042
     # 0.99250528 0.99252416 0.99255453 0.99257177 0.99260724 0.99261519
     # 0.99261519 0.99262224 0.99265349 0.99267319 0.9926738  0.99268536
     # 0.99283332 0.99283541 0.9929393  0.99304506 0.99305765 0.99315672
     # 0.9931917  0.99319538 0.99320611 0.99328682 0.99335604 0.99343061
     # 0.9934805  0.99351205 0.99356077 0.99362013 0.99366813 0.9937377
     # 0.99383332 0.99385605 0.99386609 0.99391089 0.99404144 0.99406338
     # 0.99410951 0.99410951 0.99417511 0.99419961 0.99421772 0.99425752
     # 0.99427782 0.99429702 0.99442166 0.99449317 0.9945219  0.99453681
     # 0.99470337 0.99470641 0.99472474 0.99474739 0.99477774 0.99481935
     # 0.99488838 0.99489951  
     
     # 0.99507851 0.99510856 0.99511505 0.99909081 0.99921592 0.99946948
     # 0.99514837 0.99515329 0.99518743 0.99528064 0.99528296 0.99533047
     # 0.99533426 0.99533426 0.99534394 0.99542123 0.99547203 0.99548985
     # 0.99551798 0.99563379 0.99572382 0.99572768 0.99576076 0.99577621
     # 0.99579779 0.99589647 0.99591019 0.99598953 0.99612642 0.99613569
     # 0.9961427  0.99614966 0.99615656 0.99620715 0.99625674 0.99625969
     # 0.99633017 0.99633234 0.99636056 0.9963721  0.99645986 0.9966648
     # 0.99671046 0.9967316  0.99679061 0.99684945 0.99689952 0.99690237
     # 0.99690237 0.99693057 0.99703088 0.99710978 0.99714719 0.99716166
     # 0.997221   0.997221   0.99724427 0.99726471 0.99727048 0.99728556
     # 0.99730441 0.99730441 0.99731495 0.99737073 0.99738448 0.99742312
     # 0.99749648 0.99750605 0.9975141  0.99753263 0.9975331  0.99755603
     # 0.99757893 0.99768528 0.99777454 0.9978413  0.99787139 0.99788356
     # 0.9978973  0.99795652 0.9980062  0.9980153  0.99802682 0.99813948
     # 0.99820067 0.99827398 0.9983634  0.99850547 0.99859447 0.9986378
     # 0.99870282 0.99885974 0.99886419 0.99891431 0.99891968 0.99909081
     



    


#----------------------------------------------------------------------------------
#在测噪声数据时，将以下代码粘贴到 console 更改 env.noise_a 的值就可以得到在对应噪声环境下测试保真度
#而不必再重新训练网络，可以节省大量的时间

# env.noise_a = 0
# print('\nnoise_a =',env.noise_a)
# env.noise = env.noise_a * env.noise_normal
# test_fidelity_list = testing_noise_J() #测试
# print('\nmean_test_fidelity:\n',np.mean(test_fidelity_list))

# 或者：
# h_s_list=[]
# J_s_list=[]
# h_list=[]
# J_list=[]
# for i in range(-10,11):
#     env.noise_a = i*0.01
#     print('\nnoise_a =',env.noise_a)
#     test_fidelity_list = testing_noise_h_s() #测试
#     print('mean_test_fidelity_h_s:\n',np.mean(test_fidelity_list))
#     h_s_list.append(np.mean(test_fidelity_list))
# for i in range(-10,11):
#     env.noise_a = i*0.01
#     print('\nnoise_a =',env.noise_a)
#     test_fidelity_list = testing_noise_J_s() #测试
#     print('mean_test_fidelity_J_s:\n',np.mean(test_fidelity_list))
#     J_s_list.append(np.mean(test_fidelity_list))
# for i in range(0,11):
#     env.noise_a = i*0.1
#     print('\nnoise_a =',env.noise_a)
#     test_fidelity_list = testing_noise_h() #测试
#     print('mean_test_fidelity_h:\n',np.mean(test_fidelity_list))
#     h_list.append(np.mean(test_fidelity_list))
# for i in range(0,11):
#     env.noise_a = i*0.1
#     print('\nnoise_a =',env.noise_a)
#     test_fidelity_list = testing_noise_J() #测试
#     print('mean_test_fidelity_J:\n',np.mean(test_fidelity_list))
#     J_list.append(np.mean(test_fidelity_list))
# #----------------------------------------------------------------------------------
    