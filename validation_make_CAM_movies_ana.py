from viz import CAM_movie_anasthesia, CAM_movie_None, CAM_movie_Sal
import numpy as np


thresh = 0.15

CAMS = np.load('anasthesia_results/validation_CAMS60.npy',allow_pickle='TRUE').item()

none_CAMS_all = CAMS['None']
Sal_CAMS_all = CAMS['Saline']

k = len(list(none_CAMS_all.keys()))

#%%

for i in range(k):
    none_test_idxs = list(none_CAMS_all.keys())[i]
    sal_test_idxs = list(Sal_CAMS_all.keys())[i]
    
    NoneCAMS = none_CAMS_all[none_test_idxs][0]
    SalCAMS = Sal_CAMS_all[sal_test_idxs][0]
    
    CAM_movie_anasthesia(NoneCAMS, SalCAMS, savename = 'anasthesia_results/CAM_movies/thresh_' + str(thresh) + '_' + none_test_idxs + '_' + sal_test_idxs, threshold= thresh)
    
#%%

none_avg_CAMs = np.zeros((60,91,128))


for key in list(none_CAMS_all.keys()):
    
    none_avg_CAMs += none_CAMS_all[key][0]
    

none_avg_CAMs = 1/k * none_avg_CAMs

thresh = 0

CAM_movie_None(none_avg_CAMs, savename = 'anasthesia_results/CAM_movies/avg_CAM_thresh_' + str(thresh), threshold= thresh)

#%%
sal_avg_CAMs = np.zeros((60,91,128))


for key in list(Sal_CAMS_all.keys()):
    
    sal_avg_CAMs += Sal_CAMS_all[key][0]
    

sal_avg_CAMs = 1/k * sal_avg_CAMs

thresh = 0.15

CAM_movie_Sal(sal_avg_CAMs, savename = 'anasthesia_results/CAM_movies/Sal_avg_CAM_thresh_' + str(thresh), threshold= thresh)

#%%
thresh = 0.02

none_avg_CAMs = np.zeros((60,91,128))
sal_avg_CAMs = np.zeros((60,91,128))

for i in range(k):
    none_test_idxs = list(none_CAMS_all.keys())[i]
    sal_test_idxs = list(Sal_CAMS_all.keys())[i]
    
    none_avg_CAMs += none_CAMS_all[none_test_idxs][0]
    sal_avg_CAMs += Sal_CAMS_all[sal_test_idxs][0]
    
none_avg_CAMs = 1/k * none_avg_CAMs
sal_avg_CAMs = 1/k * sal_avg_CAMs
    
CAM_movie_anasthesia(none_avg_CAMs, sal_avg_CAMs, savename = 'anasthesia_results/CAM_movies/Average_thresh_' + str(thresh), threshold= thresh)
    
