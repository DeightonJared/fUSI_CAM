from viz import CAM_movie, CAM_movie_MK,CAM_movie_Sal
import numpy as np

CAMS = np.load('mk_results/mk1_5/validation_CAMS60.npy',allow_pickle='TRUE').item()

mk_CAMS_all = CAMS['MK']
Sal_CAMS_all = CAMS['Saline']

k = len(list(mk_CAMS_all.keys()))

#%%

thresh = 0.2

for i in range(k):
    mk_test_idxs = list(mk_CAMS_all.keys())[i]
    sal_test_idxs = list(Sal_CAMS_all.keys())[i]
    
    mkCAMS = mk_CAMS_all[mk_test_idxs][0]
    SalCAMS = Sal_CAMS_all[sal_test_idxs][0]
    
    CAM_movie(mkCAMS, SalCAMS, savename = 'mk_results/mk1_5/CAM_movies/thresh_' + str(thresh) + '_' + mk_test_idxs + '_' + sal_test_idxs, threshold= thresh)
    
#%%

mk_avg_CAMs = np.zeros((60,91,128))


for key in list(mk_CAMS_all.keys()):

    
    mk_avg_CAMs += mk_CAMS_all[key][0]
    
mk_avg_CAMs = 1/k * mk_avg_CAMs

thresh = 0.2

CAM_movie_MK(mk_avg_CAMs, savename = 'mk_results/mk1_5/CAM_movies/avg_CAM_thresh_' + str(thresh), threshold= thresh)

#%%
sal_avg_CAMs = np.zeros((60,91,128))


for key in list(Sal_CAMS_all.keys()):

    
    sal_avg_CAMs += Sal_CAMS_all[key][0]
    

sal_avg_CAMs = 1/k * sal_avg_CAMs

thresh = 0.3

CAM_movie_Sal(sal_avg_CAMs, savename = 'mk_results/mk1_5/CAM_movies/Sal_avg_CAM_thresh_' + str(thresh), threshold= thresh)

#%%
thresh = 0.02

mk_avg_CAMs = np.zeros((60, 91, 128))
sal_avg_CAMs = np.zeros((60, 91, 128))

for i in range(k):
    mk_test_idxs = list(mk_CAMS_all.keys())[i]
    sal_test_idxs = list(Sal_CAMS_all.keys())[i]
   
    mk_avg_CAMs += mk_CAMS_all[mk_test_idxs][0]
    sal_avg_CAMs += Sal_CAMS_all[sal_test_idxs][0]

mk_avg_CAMs = 1/k * mk_avg_CAMs
sal_avg_CAMs = 1/k * sal_avg_CAMs
    
CAM_movie(mk_avg_CAMs, sal_avg_CAMs, savename = 'mk_results/mk1_5/CAM_movies/Average_thresh_' + str(thresh), threshold= thresh)
    


