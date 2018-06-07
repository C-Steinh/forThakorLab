import sys
import os
from os.path import dirname, abspath
sys.path.insert( 0, dirname( dirname( abspath( __file__ ) ) ) )

from mite.filters import TimeDomainFilter, JointPositionFilter, MinMaxFilter
from mite.classifiers import *
from mite.utils.Metrics import confusion_matrix

import pickle
import time
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

def rms_meas(interped, real):
    """ RMS error between interpolated and real result"""
    rms_per_E = np.zeros(8)
    rms_tot = np.zeros(8)
    rms_full = np.sqrt(np.mean((interped - real)**2))
    
    for numE in range(0,8):
        rms_per_E[numE] = (np.sqrt(np.mean((interped[:,numE] - real[:,numE])**2)))
        rms_tot[numE] = (np.sqrt(np.mean((real[:,numE])**2)))
       # plt.figure()
       # plt.ioff()
       # plt.plot(interped[:,numE],color='red',label='Interp')
       # plt.plot(real[:,numE],color='blue',label = 'Real')
       # plt.show()

    return rms_per_E, rms_tot

def interp_elecs(E0, E1, P0, P1, P_new):
    """ when the two closest electrodes are found, makes a guess about what the electrode value will ,be between the two"""
    w_E1 = abs(float(P0)-float(P_new))/abs(float(P0)-float(P1))
    w_E0 = abs(float(P1)-float(P_new))/abs(float(P0)-float(P1))
    #print(E1[:,nEs])
    #print(res)
   
    interp_value = np.multiply(w_E1,E1) + np.multiply(w_E0,E0)
    return interp_value
         
def openN(TRAINING_DATA):
    """ opens the file names above and makes readable by grasp type"""
    #print( 'Loading training data...' )
    data = pickle.load( open( TRAINING_DATA, 'rb' ) )
      
    grasps = [ 'rest', 'open', 'power', 'tripod', 'pronate', 'supinate' ]
    #current data only contains one 'trial' with several grasps in it
    #process_data(grasps,data)
    for trial in data:                              # for each trial
        return trial

################################################################################################
#################################### The main function #########################################

def main():
    relv_files = []
    #find all files in the directory
    dirFiles = os.listdir('.')
    for files in dirFiles:
        if 'train' in files:
            relv_files.append(files)
    relv_files = (sorted(relv_files))
    shift_amounts =[0, 1, 2, 3, 4, 5, 6, 7, 8, 8.5] #tests performed  - 4/19

    # grasps used
    grasps = [ 'rest', 'open', 'power', 'tripod', 'pronate', 'supinate' ]

    #find most similar electrodes later. first show that can interpolate and get something close.

    test_pos = [1, 3, 5, 7] # positions
    pair_Es = [0, 2, 4, 6] # all electrode combos that can be used shifting around the arm
    #pairs (e.g. [0,2], [2,4],[4,6],[6,0])
    
    for cur_test in range(0,len(test_pos)):
        #comparison data (test if interpolation works)
        test_DATA = relv_files[test_pos[cur_test]]
        test_trial = openN(test_DATA)

        print('test ',str(test_pos[cur_test]),': ','\t'.join(grasps))

        for cur_pair in range(0, len(pair_Es)):
            if cur_pair < (len(pair_Es)-1):
                E0_DATA = relv_files[pair_Es[cur_pair]]
                E1_DATA = relv_files[pair_Es[cur_pair+1]]
                prnt_pair = [str(pair_Es[cur_pair]),str(pair_Es[cur_pair+1])]
            else:
                E0_DATA = relv_files[pair_Es[cur_pair]]
                E1_DATA = relv_files[pair_Es[0]]
                prnt_pair = [str(pair_Es[cur_pair]),str(pair_Es[0])]

            #get the internal data
            E0_trial = openN(E0_DATA)
            E1_trial = openN(E1_DATA)

            real_res_RMS = []
            interp_res_RMS= []
            for grasp in grasps:
        
                #interpolate (use function for each grasp)
                E0_cur = E0_trial[grasp]['MyoArmband']
                E1_cur = E1_trial[grasp]['MyoArmband']
                dim_1 = min(E0_cur.shape[0],min(E1_cur.shape[0],test_trial[grasp]['MyoArmband'].shape[0]))
                interp_res = interp_elecs(E0_cur[:dim_1,:], E1_cur[:dim_1,:], 0, 2, 1) # currently true for all of these
    
                #rms of the interpolated and real
                RMS_interped, RMS_tot = rms_meas(interp_res,test_trial[grasp]['MyoArmband'][:dim_1,:])
                interp_res_RMS.append(str(np.mean(RMS_interped)))
                real_res_RMS.append(str(np.mean(RMS_tot)))
            print('int', ''.join(prnt_pair),': ','\t'.join(interp_res_RMS))
        print('real : ', '\t'.join(real_res_RMS))

    
        #plotting all the figures
        #plt.figure()
        # plt.ioff()
        #fig, axes = plt.subplots(nrows = 4, ncols= 2, figsize = (7,7))
        #for numElec in range(0,8):
        #    ax = plt.subplot(4,2,numElec+1)
    
        #    plt.plot(test_trial[grasp]['MyoArmband'][:,numElec],color = 'blue')
        #    plt.plot(interp_res[:,numElec],color = 'red')
        #    ax.set_title(''.join(['Elec: ',str(numElec)]))
            
        #    fig.suptitle(''.join([grasp, ' Interp Test']))
        #plt.savefig(''.join(['interp_E_test_',grasp,'_1']))
        #plt.close('all')
                 
            
if __name__ == '__main__':
    main()
