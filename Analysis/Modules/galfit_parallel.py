# https://github.com/Grillard/GalfitPyWrap may be useful for setting up inputs
from __future__ import division
import sys
sys.path.insert(0,'/data/emiln/XLSSU122/analysis/galfit/GalfitPyWrap')
from GalfitPyWrap import galfitwrap as gf
import os
import numpy as np
import pandas as pd
import glob
from scipy import ndimage
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io import ascii
from astropy import wcs
from astropy.table import Table, hstack, join
import os

import multiprocessing as mp

def run_galfit_cosmos_parallel(row1,OG_df=cos_df_OG,zp=26,width=90,PSFf=1,use_psf=True,psf_file='tinytim_psf.fits',sigma_file='/sigma_meanexp_cutout.fits',save_name='rmssigmameanexp_w180_TESTING'):
    cutout_width = 200 # ALWAYS KEEP THE SAME, PHYSICAL SIZE OF CUTOUTS FROM DATA_PREP
    N = 0 # Extra width to search for neighbours in?
#     zp = 26 # http://www.stsci.edu/hst/wfc3/analysis/ir_phot_zpt

    res = []
    re_errs = []
    ars = []
    ar_errs = []
    ns = []
    n_errs = []
    ids_all = []
    oimg_all = []

    df = OG_df
    r = row1
    ra = r.RA
    dec = r.DEC
    pixcrd = full_wcs.wcs_world2pix(ra,dec, 1)
    print pixcrd
    X = int(pixcrd[0])
    Y = int(pixcrd[1])
    ID = int(r['NUMBER'])
#     ids_all.append(ID)
    CX = cutout_width
    CY = cutout_width

    print "ID", ID
    print "RA:",ra
    print "DEC:",dec
    print "Initial X:", X
    print "Initial Y:", Y
    print "Cutout X:", CX 
    print "Cutout Y:", CY

    tdir = '/data/emiln/XLSSU122/analysis/cosmos/galfit_results/'+str(ID)

    model=[{
    0: 'sersic',              #  object type
    1: str(CX)+' '+str(CY)+' 1 1', #  position x, y
    3: '21 1',            #  Integrated magnitude   
    4: '10 1',            #  R_e (half-light radius)   [pix]
    5: '4 1',            #  Sersic index n (de Vaucouleurs n=4) 
    9: '1 1',            #  axis ratio (b/a)  
    10: '0 1',         #  position angle (PA) [deg: Up=0, Left=90]
    'Z': 0                   #  output option (0 = resid., 1 = Don't subtract) 
    }]

    bounds = [CX-width,CX+width,CY-width,CY+width]
    bounds2 = [X-width,X+width,Y-width,Y+width]
    print "Bounds:", bounds

    # Find neighbours in full DF
    # Neighbour check needs to be done on original fits image / catalog
    ndf = df[(df['X']>(bounds2[0]-N)) & (df['X']<(bounds2[1]+N)) & (df['Y']>(bounds2[2]-N)) & (df['Y']<(bounds2[3]+N))
             & (df['NUMBER']!=ID)]

    print len(ndf),"NEIGHBOURS FOUND"

    print "Adding additional model components for neighbours..."
    for row in ndf.iterrows():
        r = row[1]
        NX = r.X - X + cutout_width
        NY = r.Y - Y + cutout_width
        seqnr = int(r['NUMBER'])
        model.append({
            0: 'sersic',              #  object type
            1: str(NX)+' '+str(NY)+' 1 1', #  position x, y
            3: '21 1',            #  Integrated magnitude   
            4: '10 1',            #  R_e (half-light radius)   [pix]
            5: '4 1',            #  Sersic index n (de Vaucouleurs n=4) 
            9: '1 1',            #  axis ratio (b/a)  
            10: '0 1',         #  position angle (PA) [deg: Up=0, Left=90]
            'Z': 0                   #  output option (0 = resid., 1 = Don't subtract) 
            })

    if use_psf:
        print "Using PSF"
        O=gf.CreateFile(tdir+'/cutout.fits', bounds, model,fout=tdir+'/input.feedme',\
                        Pimg=psf_file, Simg=tdir+sigma_file, ZP=zp, scale='0.06 0.06',PSFf=PSFf)
    else:
        O=gf.CreateFile(tdir+'/cutout.fits', bounds, model,fout=tdir+'/input.feedme',\
                        Simg=tdir+sigma_file, ZP=zp, scale='0.06 0.06')

    p,oimg,mods,EV,chi2nu=gf.rungalfit(tdir+'/input.feedme',verb=False)
    bad_result = False
    try: 
        os.rename('out.fits',tdir+'/out_'+save_name+'.fits')
        os.rename('fit.log',tdir+'/fit_'+save_name+'.log')
        os.rename('galfit.01',tdir+'/galfit_'+save_name+'.01')
        for m in mods[0]:
            print m, mods[0][m]
            if "*" in mods[0][m] and "MAG" not in m:
                bad_result = True
                print "***Bad Result***"
                break
    except:
        print "***Fit failed***"
        

    if chi2nu > 100:
        print "***Bad Result, chi2nu>100***"
        bad_result=True

    try:
        if bad_result == False:
            re_all = mods[0]['1_RE'].split(' ')
            ar_all = mods[0]['1_AR'].split(' ')
            n_all = mods[0]['1_N'].split(' ')
            re = float(re_all[0])
            re_err = float(re_all[2])
            ar = float(ar_all[0])
            ar_err = float(ar_all[2])
            n = float(n_all[0])
            n_err = float(n_all[2])
        if bad_result == True:
            re_all = [-99,-99,-99]
            ar_all = re_all
            n_all = re_all
            re = float(re_all[0])
            re_err = float(re_all[2])
            ar = float(ar_all[0])
            ar_err = float(ar_all[2])
            n = float(n_all[0])
            n_err = float(n_all[2])
    except:
        re_all = [-99,-99,-99]
        ar_all = re_all
        n_all = re_all
        re = float(re_all[0])
        re_err = float(re_all[2])
        ar = float(ar_all[0])
        ar_err = float(ar_all[2])
        n = float(n_all[0])
        n_err = float(n_all[2])

#     res.append(re)
#     re_errs.append(re_errs)

#     ars.append(ar)
#     ar_errs.append(ar_errs)

#     ns.append(n)
#     n_errs.append(n_errs)

#     data = {'ID': ids_all, 're': res, 'ar': ars, 'n': ns}
#     save_df = pd.DataFrame(data,columns=['ID','re','ar','n'])
#     save_df.to_csv('COSMOS_test_galfit_results_'+save_name+'.csv',index=False)
#     pickle.dump(oimg_all, open())
    return re, ar, n