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

def make_feats(trial,grasps):
    WINDOW_SIZE = 10                                # in samples
    WINDOW_STEP = 2

    #print( 'Creating filters...' )
    td5 = TimeDomainFilter( num_features = 5 )
    
    key_count = 0
    X = []
    y=  []

    for grasp in grasps:
        emg = trial[ grasp ][ 'MyoArmband' ][ :, :8 ]  # grab the raw EMG data
        # calculating stuff
        feat = []
        label = []
 
        n_samples = emg.shape[0]
      
        for sample in range(WINDOW_SIZE,n_samples,WINDOW_STEP):
            #for MAV analysis
            # round(round(0.35*n_samples),-1),round(round(0.65*n_samples),-1), WINDOW_STEP ):

            #feat.append( td5.filter( emg[sample-WINDOW_SIZE:sample, :] ))
            #MAV_Es = np.zeros(8)
            #for nEs in range(0,8):
            #    MAV_Es[nEs] =(np.mean(abs( emg[sample-WINDOW_SIZE:sample, nEs] )))
            feat.append(td5.filter( emg[sample-WINDOW_SIZE:sample, :]))               # calculate features
                
            #feat.append(MAV_Es)
            label.append( key_count )
    
        X.append( np.vstack( feat ) )
        y.append( np.vstack( label ) )

        key_count += 1

    # these are now matrices
    X = np.vstack( X )
    y = np.squeeze( np.vstack( y ) )#labels which gesture it the window is from
    
    #print( '\nPreprocess data...', end = ' ', flush = True )
    
    #zscore the data - keeps variance the same
    mmfilter = MinMaxFilter( X, dynamic = True )
    X = mmfilter.filter( X ) # all the original train data is train data for this new test run

    return X,y

def classify_data_w_shift(TRAINING_DATA, TEST_DATA, grasps, shift):

    train_trial = TRAINING_DATA
    test_trial = TEST_DATA

    MODEL = 'SRC'

    #shift Xtrain, ytrain should remain the same b/c the gesture order is still the same

    #make feature vectors for test and training data
    #print(shift)
     # print( grasp, '...', end = ' ', flush = True )

    for grasp in grasps:                        # for each grasp
        emg_test = test_trial[ grasp ][ 'MyoArmband' ][ :, :8 ]
       # print(emg_test[:5,:])
        test_trial[grasp]['MyoArmband'] = np.roll(emg_test,shift, axis =1)
       # print(test_trial[grasp]['MyoArmband'][:5,:])
    Xtrain, ytrain = make_feats(train_trial,grasps)
    Xtest, ytest = make_feats(test_trial,grasps)
     
    # use the original train data on the new test data but shifted!
    #try split
    Xtrain, Xble,ytrain,ybleh = train_test_split(Xtrain,ytrain,test_size = 0.33)

    #print('Creating classifiers...', end= ' ', flush = True)
    mdl = eval( MODEL )( Xtrain, ytrain)

    #print('Testing classifier...', end = ' ', flush = True)
    yhat, yprob = mdl.predict( Xtest, prob = True )
   

    print('Classifier done!')

    class_labels =[ 'RE', 'HO', 'HC', 'TR', 'WP', 'WS' ]
    cm = confusion_matrix( ytest, yhat, labels = class_labels, cmap = 'Blues' )
    return cm, yhat, ytest

    
def find_location(CANON_trial,test_trial,grasps,text_file,shift_amounts,real_shift):
    doplotnsave = 1
    
    all_shifts = []
      # pronate, as assumed to be the most uni-directional an distinct is used for registration (hypothesis - for original data open and tripod both look pretty good - going to use that)

    for grasp in grasps: # #1,3
        best_shifts = []
    
        #canonical position (used for finding rotation)

        # get static measure of EMG (mean-absolute value of signal)
        CANON_emg = CANON_trial[grasp]
        CANON_val = mean_abs(CANON_emg,grasp)

        if doplotnsave:
            #text_file.write(''.join([grasp,'\n']))
            #plotting all the figures
            plt.ioff()
            fig, axes = plt.subplots(nrows = 1, ncols= 2, figsize = (7,7))
           # ax = plt.subplot(1,2,1)
           # plt.plot(CANON_val, color = 'blue')
           # ax.set_title('CANON')

            #plot and show shifted mean results for a grasp, indicating it is a consistent pattern
    
        #use RMS to guess what rotation each position is at.
        #comparison data (test if finding rotation works)
        test_emg = test_trial[grasp]
        val_per_curtest = mean_abs(test_emg,grasp)

        #shift cannon to find best rms
        if doplotnsave:
            ax = plt.subplot(1,2,2)
            real, = plt.plot(val_per_curtest, color = 'blue',label = 'Real Trace')       
            shifted, = plt.plot(np.roll(CANON_val,real_shift),color='red', label = 'Shifted')
        pred_rmses = []

        for numShifts in range(0,8):#number of possibl positions
            #shifted version of original and the test ( in terms of windowed mean absolute value function
            pred_rms, real_rms= rms_meas(np.roll(CANON_val,numShifts),val_per_curtest)
            pred_rmses.append(pred_rms)
        print(pred_rmses)

       # if doplotnsave:
                # writes information for each position and test rms into a text file
                #text_file.write(''.join(['Current Test: ', str(cur_test),' RMS: ',str(real_rms),'\n']))
                #text_file.write(''.join([str(pred_rmses),'\n']))
        best_shift = pred_rmses.index(min(pred_rmses))

        print(best_shift)
        best_shifts.append(best_shift)
        if doplotnsave:
             #   text_file.write(''.join([str(best_shift),'\n']))
                # chose the best prediction as shift by lowest rms value
            best_pred, = plt.plot(np.roll(CANON_val,best_shift),color='green',label = 'Best RMS')
            plt.legend(handles = [real,shifted, best_pred])
            ax.set_title(''.join(['Shift by ',str(real_shift),' Best:', str(best_shift) ]))
            #title to the figure when printing a figure with subplots of the results
      
        if doplotnsave:
            fig.suptitle(''.join(['Mean Absolute Value per Position (', grasp,')']))
          #  fig.savefig(''.join([grasp,'_best_res.png']))
            plt.show()
        all_shifts.append(best_shifts)
 
    # list of lists  of all the necessary shifts
    return all_shifts

def mean_abs(curemg,grasp):
    WINDOW_SIZE = 10
    WINDOW_STEP = 2 # does not effect results much

    n_samples = curemg.shape[0]
     
    val_per_E = []
    for nE in range(0,8):
        feat = []
        for sample in range(WINDOW_SIZE,n_samples,WINDOW_STEP):
            
            #feat.append(td5.filter(emg[sample-WINDOW_SIZE:sample,:]))
            feat.append(np.mean(abs( curemg[sample-WINDOW_SIZE:sample,nE])))
        temp_feat = feat
        if nE is 0:
            feat_fin = np.array(feat)
            #flex_rng = [round(0.35*feat_fin.shape[0]), round(0.65*feat_fin.shape[0])]
            #get dimensions for finding flexing zone
            val_per_E = [(np.mean(feat_fin))]# now giving just a subsection instead of this: [range(flex_rng[0],flex_rng[1])]))]
        else:
            feat_fin = np.vstack((feat_fin, np.array(feat)))
            val_per_E.append(np.mean(feat_fin[nE,:]))#range(flex_rng[0],flex_rng[1])]))
    return val_per_E
    
def rms_meas(pred, real):
    """ RMS error between interpolated and real result"""
    rms_pred_real = np.zeros(8)
    rms_tot = np.zeros(8)
    rms_tot = np.sqrt(np.mean((np.array(real))**2))
    rms_pred_real = (np.sqrt(np.mean((pred - real)**2)))

    return rms_pred_real, rms_tot

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

    curdir = '.'#'/home/cynthia/forThakorLab/chris_train_6_7_18'
    text_file = open('output.txt','w')
    relv_files = []
    #find all files in the directory
    dirFiles = os.listdir(curdir)
    for files in dirFiles:
        if 'train_' in files:
            relv_files.append(files)
    relv_files = (sorted(relv_files))
    # 0,1,2,3,3,4,5,6,7,7.5
    #4/19 shift_amounts = [0,1,2,3,3,4,5,6,7]#
    shift_amounts = [0, 1, 2, 3 ,3, 4, 5, 6, 7] #tests performed  - 4/19

    relv_files = relv_files[:9]

    # grasps used
    grasps =  ['rest', 'open', 'power', 'tripod', 'pronate', 'supinate' ]
    # [ 'rest', 'open', 'power', 'tripod', 'pronate', 'supinate' ]

    #change to data directory:
    os.chdir(curdir)
    q= os.getcwd()
    print(q)

    #rng where subject most likely flexing
    full_basis = []
    test_samples = []
    shift_samples = []
    for cur_shift in range(0,len(relv_files)):
        cur_basis = openN(relv_files[cur_shift])
        #breaking up the current electrode into a shift, train and test set
        shift_data = dict()
        train_data = dict()
        test_data = dict()
        
        for grasp in grasps:
            sess = cur_basis[grasp]['MyoArmband'][:,:8]#just EMG electrodes
            flex_rng = [round(0.35*sess.shape[0]), round(0.65*sess.shape[0])]
            #print(flex_rng)
            trial_split = range(flex_rng[0],flex_rng[1],round((flex_rng[1]-flex_rng[0])/3))
            shift_data[grasp]= (sess[range(trial_split[0],(trial_split[1]-1)),:8])
            train_data[grasp] = (sess[range(trial_split[1],(trial_split[2]-1)),:8])
            test_data[grasp]= (sess[range(trial_split[2],(flex_rng[1])),:8])

        shift_samples.append(test_data)
        test_samples.append(test_data)
        full_basis.append(train_data)

    #now that data is split try to predict it
    best_pos = find_location(shift_samples[0],shift_samples[5],grasps,text_file,shift_amounts,shift_amounts[5])
    print(best_pos)

    #makes training set of each of the shifts to use when trying to predict
    #tripod was distinctive so use tripod (3) as a predictor
    curshift = 6
    pre_shift = best_pos[3][curshift]
    #know 0:7 are the shifts we have as a basis. make training data from that.
    # restrict to the 2 s mark up to the 4 s mark of training
    make_feats(trial,grasps)
    
    #classification using the old classifier with shifted features
    #prints out increase and classification of each class as percentage across shifts

 #   # for nshifts in range(0,len(shift_amounts)):
 #   for pred_num in range(0,len(relv_files)-1):
 #       print(str(pred_num))
        #q= openN(relv_files[pred_num])
 #       cm_shift, yh_shift, yt_shift = classify_data_w_shift(openN(relv_files[0]),openN(relv_files[pred_num]), grasps,-shift_amounts[pred_num])
        
 #       cm_ns, yh_shift, yt_shift = classify_data_w_shift(openN(relv_files[0]),openN(relv_files[pred_num]), grasps,0)
       
 #       shft_res = np.zeros(5)
 #       for n in range(0,5):
 #           shft_res[n] = cm_shift[n,n]-cm_ns[n,n]
 #           print(str(cm_shift[n,n]-cm_ns[n,n]))

 #       print(str(np.mean(shft_res)),str(np.std(shft_res)))
       #best_pos[4][2])
      
if __name__ == '__main__':
    main()
