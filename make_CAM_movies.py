from viz import get_CAMs, plot_CAMs, showImg, showImgs, CAM_movie, showImgCAM, mouse_movie, CAM_movie_cropped, CAM_movie_MK, CAM_movie_MK_cropped
import numpy as np

# mkCAMS = np.load('MKCAMS_1m.npy')

# SalCAMS = np.load('SalCAMS_1m.npy')

CAMS = np.load('mk_results/mk1_5/validation_CAMS60.npy',allow_pickle='TRUE').item()

mk_CAMS_all = CAMS['MK']
Sal_CAMS_all = CAMS['Saline']

#mkCAMS = np.load('05_MKCAMS_1m.npy')
# SalCAMS = np.load('05_SalCAMS_1m.npy')

mkCAMS = mk_CAMS_all[list(mk_CAMS_all.keys())[8]][0]
SalCAMS = Sal_CAMS_all[list(Sal_CAMS_all.keys())[8]][0]

thresh = 0.2

CAM_movie(mkCAMS, SalCAMS, savename = '05_best_performing_thresh_' + str(thresh), threshold= thresh)

# CAM_movie_cropped(mkCAMS, SalCAMS, savename = '0322/mc_0/old/thresh_' + str(thresh), threshold= thresh)

# CAM_movie_MK_cropped(mkCAMS, savename= 'CroppedResults/nonrigid_new/thresh_' + str(thresh) , threshold= thresh) 