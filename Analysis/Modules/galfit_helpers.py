from __future__ import division
import sys
sys.path.insert(0,'/data/emiln/XLSSC122_GalPops/Analysis/Modules')
from GalfitPyWrap import galfitwrap as gf
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from astropy.table import Table, hstack, join
import fnmatch
import multiprocessing as mp

def run_galfit_cosmos_parallel(row1,OG_df=cos_df_OG,zp=26,width=90,HLwidth=False,PSFf=1,use_psf=True,\
                               timeout=300,verb=False,psf_file='tinytim_psf.fits',\
                               sigma_file='/sigma_meanexp_cutout.fits',cutout_file='/cutout.fits',\
                               save_name='rmssigmameanexp_w180_TESTING',\
                              PA_INIT=45, AR_INIT=0.5, MAG_INIT=21,DYNMAG=False, convbox='100 100',
                              constr='none',cutout_width = 200, badmask='none', MAGFIT=False, mag_thresh=5,
                               sky='Default',skyINIT=0, N=0):
#     cutout_width = 200 # ALWAYS KEEP THE SAME, PHYSICAL SIZE OF CUTOUTS FROM DATA_PREP
#     N = 0 # Extra width to search for neighbours in?
#     mag_thresh = 4 # Neighbours only fit if less than mag_thresh fainter than primary
    
    df = OG_df
    r = row1
    ra = r.RA
    dec = r.DEC
    ID = int(r['NUMBER'])
    if badmask != 'none':
        print "Using bad pixel mask..."
        badmask = tdir+badmask
    if DYNMAG:
        MAG_INIT=np.round(r.F125W_Kron,decimals=2)
        og_mag=MAG_INIT
        print "Initializing",str(ID),"with F125W Kron magnitude:", MAG_INIT
    pixcrd = full_wcs.wcs_world2pix(ra,dec, 1)
    print pixcrd
    X = int(pixcrd[0])
    Y = int(pixcrd[1])
    
#     ids_all.append(ID)
    CX = cutout_width
    CY = cutout_width
    
    if MAGFIT:
        magf = 1
    else:
        magf = 0

    print "ID", ID
    print "RA:",ra
    print "DEC:",dec
    print "Initial X:", X
    print "Initial Y:", Y
    print "Cutout X:", CX 
    print "Cutout Y:", CY
    print "Cutout width:", cutout_width

    # tdir = '/data/emiln/XLSSU122/analysis/cosmos/galfit_results/'+str(ID)
    tdir = '/data/emiln/XLSSC122_GalPops/Data/Products/COSMOS/galfit_results/'+str(ID)
    
    model=[{
    0: 'sersic',              #  object type
    1: str(CX)+' '+str(CY)+' 1 1', #  position x, y
    3: str(MAG_INIT)+' '+str(magf),            #  Integrated magnitude   
    4: '10 1',            #  R_e (half-light radius)   [pix]
    5: '4 1',            #  Sersic index n (de Vaucouleurs n=4) 
    9: str(AR_INIT)+' 1',            #  axis ratio (b/a)  
    10: str(PA_INIT)+' 1',         #  position angle (PA) [deg: Up=0, Left=90]
    'Z': 0                   #  output option (0 = resid., 1 = Don't subtract) 
    }]

    if HLwidth == False:
        bounds = [CX-width,CX+width,CY-width,CY+width]
        bounds2 = [X-width,X+width,Y-width,Y+width]
        print "Cutoutwidth (pixels):", width*2
        print "Cutoutwidth (arcsec):", width*2*0.06
    else: 
        width = int(np.ceil(HLwidth*r.HLR))
        bounds = [CX-width,CX+width,CY-width,CY+width]
        bounds2 = [X-width,X+width,Y-width,Y+width]
        print "Cutoutwidth (pixels) for ID",str(ID),":", width*2
        print "Cutoutwidth (arcsec) for ID",str(ID),":", width*2*0.06
        
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
        if DYNMAG:
            MAG_INIT=np.round(r.F125W_Kron,decimals=2)
            if np.isnan(MAG_INIT):
                MAG_INIT=og_mag # Initialize with MAG of primary target
                print "***NEIGHBOUR MAG NOT CATALOGED***"
            print "NEIGHBOUR initialized with F125W Kron magnitude:", MAG_INIT
            if MAG_INIT > og_mag+mag_thresh:
                print "Neighbour mag too faint, not being fit"
                continue # If mag is too faint, do not fit this object (Should eventually mask these)
                
        seqnr = int(r['NUMBER'])
        model.append({
                0: 'sersic',              #  object type
                1: str(NX)+' '+str(NY)+' 1 1', #  position x, y
                3: str(MAG_INIT)+' '+str(magf),            #  Integrated magnitude   
                4: '10 1',            #  R_e (half-light radius)   [pix]
                5: '4 1',            #  Sersic index n (de Vaucouleurs n=4) 
                9: str(AR_INIT)+' 1',            #  axis ratio (b/a)  
                10: str(PA_INIT)+' 1',         #  position angle (PA) [deg: Up=0, Left=90]
                'Z': 0                   #  output option (0 = resid., 1 = Don't subtract) 
                })


    if use_psf:
        print "Using PSF"
        O=gf.CreateFile(tdir+cutout_file, bounds, model,fout=tdir+'/input.feedme',\
                        Pimg=psf_file, Simg=tdir+sigma_file, ZP=zp, scale='0.06 0.06',PSFf=PSFf, convbox=convbox,
                       constr=constr, badmask=badmask, sky=sky,skyINIT=skyINIT)
        # convbox should be larger than PSF. f125_400 psf is (166,166)
    else:
        O=gf.CreateFile(tdir+cutout_file, bounds, model,fout=tdir+'/input.feedme',\
                        Simg=tdir+sigma_file, ZP=zp, scale='0.06 0.06', PSFf=PSFf, convbox=convbox, constr=constr,
                       badmask=badmask,sky=sky,skyINIT=skyINIT)

    p,oimg,mods,EV,chi2nu=gf.rungalfit(tdir+'/input.feedme',verb=verb, timeout=timeout)
    if EV==124: print "***PROCESS TIMEOUT***"
    bad_result = False
    print '****',ID,'****'
    try: 
        os.rename('out.fits',tdir+'/out_'+save_name+'.fits')
        os.rename('fit.log',tdir+'/fit_'+save_name+'.log')
        os.rename('galfit.01',tdir+'/galfit_'+save_name+'.01')
    except:
        print "***",ID,"Fit failed at GALFIT level***"
        

    try:
#         print "start try"
        print mods
        re_all = mods[0]['1_RE'].split(' ')
        ar_all = mods[0]['1_AR'].split(' ')
        n_all = mods[0]['1_N'].split(' ')
        m_all = mods[0]['1_MAG'].split(' ')
        s_all = mods[-1][str(len(mods))+'_SKY'].split(' ')
#         print "split('') complete"
        re_val = re_all[0].split('*')
        re_err_val = re_all[2].split('*')
        ar_val = ar_all[0].split('*')
        ar_err_val = ar_all[2].split('*')
        m_val = m_all[0].split('*')
        m_err_val = m_all[2].split('*')
        n_val = n_all[0].split('*')
        n_err_val = n_all[2].split('*')
        s_val = s_all[0].split('*')
        s_err_val = s_all[2].split('*')
#         print "split('*') complete"
        
        print re_val
        print re_err_val
        print m_val
        print m_err_val
        print ar_val
        print ar_err_val
        print s_val
        print s_err_val

        if len(re_val)>1:
#             EV+=2**2
            print 'ID',str(ID),'len(re_val)>1', len(re_val)>1

        if len(ar_val)>1:
#             EV+=2**3
            print 'ID',str(ID),'len(ar_val)>1', len(ar_val)>1

        if len(n_val)>1:
#             EV+=2**4
            print 'ID',str(ID),'len(n_val)>1', len(n_val)>1

        if len(m_val)>1:
#             EV+=2**5
            print 'ID',str(ID),'len(m_val)>1', len(m_val)>1
            
        if len(s_val)>1:
#             EV+=2**5
            print 'ID',str(ID),'len(s_val)>1', len(s_val)>1
            
#         print "EV updated for *"
        
        
#         print 'int(np.ceil(len(re_val)/2.)-1)', int(np.ceil(len(re_val)/2.)-1)
        re = float(re_val[int(np.ceil(len(re_val)/2.)-1)])
#         print 're', re
        
#         print 'int(np.ceil(len(re_err_val)/2.)-1)', int(np.ceil(len(re_err_val)/2.)-1)
        re_err = float(re_err_val[int(np.ceil(len(re_err_val)/2.)-1)])
#         print 're_err', re_err
        
#         print int(np.ceil(len(ar_val)/2.)-1)
        ar = float(ar_val[int(np.ceil(len(ar_val)/2.)-1)])
#         print ar
        ar_err = float(ar_err_val[int(np.ceil(len(ar_err_val)/2.)-1)])
#         print ar_err
        n = float(n_val[int(np.ceil(len(n_val)/2.)-1)])
#         print n
        n_err = float(n_err_val[int(np.ceil(len(n_err_val)/2.)-1)])
#         print n_err
        m = float(m_val[int(np.ceil(len(m_val)/2.)-1)])
#         print m
        m_err = float(m_err_val[int(np.ceil(len(m_err_val)/2.)-1)])
    
        s = float(s_val[int(np.ceil(len(s_val)/2.)-1)])
        s_err = float(s_err_val[int(np.ceil(len(s_err_val)/2.)-1)])

                       
    except:
        print "EXCEPTION"
        re_all = [-99,-99,-99]
        ar_all = re_all
        n_all = re_all
        m_all = re_all
        s_all = re_all
        re = float(re_all[0])
        re_err = float(re_all[2])
        ar = float(ar_all[0])
        ar_err = float(ar_all[2])
        n = float(n_all[0])
        n_err = float(n_all[2])
        m = float(m_all[0])
        m_err = float(m_all[2])
        s = float(s_all[0])
        s_err = float(s_all[2])
        EV+=999

    return ID,ra,dec, re, ar, n, m, s, chi2nu, EV

def showme3(oimg,fignum=None):
    vm = np.percentile(oimg[1].data,99)
#     print vm
    
    plt.figure(fignum)
    plt.subplot(131)
    plt.imshow(oimg[1].data,interpolation='none',cmap='viridis',vmin=1e-4,vmax=vm)
    plt.subplot(132)
    plt.imshow(oimg[2].data,interpolation='none',cmap='viridis',vmin=1e-4,vmax=vm)
    plt.subplot(133)
    plt.imshow(oimg[3].data,interpolation='none',cmap='viridis',vmin=1e-4,vmax=vm)

def plot_by_ID(ID,save_name=None,cos_df=cos_df):
    if not save_name:
        print 'provide a save_name...'
        return
    # tdir = '/data/emiln/XLSSU122/analysis/cosmos/galfit_results/'+str(ID)
    tdir = '/data/emiln/XLSSC122_GalPops/Data/Products/COSMOS/galfit_results/'+str(ID)
    
#     ttdf = pd.read_csv('COSMOS_test_galfit_results_'+save_name+'.csv')
    results_root = '/data/emiln/XLSSC122_GalPops/Analysis/COSMOS/Results/'
    # ttdf = pd.read_csv('results/'+save_name+'.csv')
    ttdf = pd.read_csv(results_root+save_name+'.csv')
    print 'ID:',ID
    print 'ttdf[ID]',ttdf['ID']
    print ttdf[ttdf['ID'] == ID]
#     print ID
    try: 
        oimg = fits.open(tdir+'/out_'+save_name+'.fits')
        showme3(oimg)
    except:
        print "No model fit found. Plotting raw image from /cutout.fits"
        try:
            oimg = fits.open(tdir+'/cutout.fits')
            vm = np.percentile(oimg[0].data,99)
            plt.figure()
            plt.subplot(131)
            plt.imshow(oimg[0].data,interpolation='none',cmap='viridis',vmin=1e-4,vmax=vm)
        except:
            print "No cutout.fits file found..."
            oimg = fits.open(tdir+'/data_cps.fits')
            vm = np.percentile(oimg[0].data,99)
            plt.figure(figsize=(5,5))
            plt.subplot()
            plt.imshow(oimg[0].data,interpolation='none',cmap='viridis',vmin=1e-4,vmax=vm)
        
    print '\n ***CATALOG PARAMS***'
    print cos_df[cos_df['NUMBER']==ID][['q','n','re']]
    plt.show()

def compare_df(galfit_file,catalog_df,sky=False):
    # Load fits and compare to published catalog
    df2 = pd.read_csv(galfit_file)
    len(df2[df2['n']==-99])
    df = df2[df2['n']!=-99]
    cos_df = catalog_df

    plt.hist(cos_df['n'],bins=np.linspace(0,10,20),alpha=0.5,color='blue', label='Catalog')
    plt.hist(df['n'],bins=np.linspace(0,10,20), alpha=0.5,color='red', label='Test')
    plt.xlabel('Sersic index, n')
    plt.legend()
    plt.show()
    print np.max(cos_df['n'])

    plt.hist(cos_df['re'],bins=np.linspace(0,6,30),alpha=0.5,color='blue', label='Catalog')
    plt.hist(df['re']*0.06,bins=np.linspace(0,6,30), alpha=0.5,color='red', label='Test')
    plt.xlabel('Effective radius (\'\')')
    plt.legend()
    plt.show()

    plt.hist(cos_df['q'],bins=np.linspace(0,1,20),alpha=0.5,color='blue', label='Catalog')
    plt.hist(df['ar'],bins=np.linspace(0,1,20), alpha=0.5,color='red', label='Test')
    plt.xlabel('Axis ratio, q')
    plt.legend()
    plt.show()
    
    mdf = df.merge(cos_df,left_on='ID',right_on='NUMBER',suffixes=('_M','_C'))
    
    print mdf.columns.values
    
    mdf['logn_M'] = np.log10(mdf['n_M'])
    mdf['logn_C'] = np.log10(mdf['n_C'])
    mdf['re_M'] = mdf['re_M']*0.06
    
    mdf['Delta_n'] = mdf['n_M'] - mdf['n_C'] 
    mdf['Delta_re'] = mdf['re_M'] - mdf['re_C']
    mdf['Delta_q'] = mdf['ar'] - mdf['q']
    mdf['Delta_mag'] = mdf['mag_M'] - mdf['mag_C'] 
    
    plt.scatter(mdf['F125W_Kron'],mdf['Delta_n'],facecolors='none',s=10,edgecolors='k')
    plt.ylabel(r'$\delta_n$')
    plt.xlabel(r'$F125W_{Kron}$')
    plt.ylim([-2,2])
    plt.show()
    
    plt.scatter(mdf['F125W_Kron'],mdf['Delta_re'],facecolors='none',s=10,edgecolors='k')
    plt.ylabel(r'$\delta_{re}$')
    plt.xlabel(r'$F125W_{Kron}$')
    plt.ylim([-0.5,0.5])
#     plt.yscale('log')
    plt.show()
    
    plt.scatter(mdf['F125W_Kron'],mdf['Delta_q'],facecolors='none',s=10,edgecolors='k')
    plt.ylabel(r'$\delta_q$')
    plt.xlabel(r'$F125W_{Kron}$')
    plt.ylim([-1,1])
    plt.show()
    
    plt.scatter(mdf['F125W_Kron'],mdf['Delta_mag'],facecolors='none',s=10,edgecolors='k')
    plt.ylabel(r'$\delta_{mag}$')
    plt.xlabel(r'$F125W_{Kron}$')
    plt.ylim([-1,1])
    plt.show()
    
#     plt.scatter(mdf['F125W_Kron'],mdf['Delta_n']/mdf['n_C'],facecolors='none',s=10,edgecolors='k')
#     plt.ylabel(r'$\delta_n/n_c$')
#     plt.xlabel(r'$F125W_{Kron}$')
#     plt.show()
    
    plt.scatter(mdf['n_M'],mdf['n_C'],facecolors='none',s=10,edgecolors='k')
    plt.plot([0,8],[0,8],'k--')
    plt.xlabel(r'Measured $n$')
    plt.ylabel(r'Catalog $n$')
    plt.xlim([0,8])
    plt.ylim([0,8])
    plt.show()
    
    print "Mean abs dex difference between measured and catalog n:", np.mean(np.sqrt((mdf['logn_M']-mdf['logn_C'])**2))
    print "Median abs dex difference between measured and catalog n:", np.median(np.sqrt((mdf['logn_M']-mdf['logn_C'])**2))
    print "Mean abs diff between measured and catalog n:", np.mean(abs(mdf['n_M']-mdf['n_C']))
    print "Median abs diff between measured and catalog n:", np.median(abs(mdf['n_M']-mdf['n_C']))
    print "Mean difference between measured and catalog n:", np.mean(mdf['n_M']-mdf['n_C'])
    print "Median difference between measured and catalog n:", np.median(mdf['n_M']-mdf['n_C'])
    print "STD of difference between measured and catalog n:", np.std(mdf['n_M']-mdf['n_C'])
    print "Median offset in delta(n)/n in dex:", np.median((mdf['logn_M']-mdf['logn_C'])/mdf['logn_C'])
    print "Scatter in in delta(n)/n in dex:", np.std((mdf['logn_M']-mdf['logn_C'])/mdf['logn_C'])
    
    
    mdf['logr_M'] = np.log10(mdf['re_M'])
    mdf['logr_C'] = np.log10(mdf['re_C'])
    
    plt.scatter(mdf['re_M'],mdf['re_C'],facecolors='none',s=10,edgecolors='k')
    plt.plot([0,8],[0,8],'k--')
    plt.xlabel(r'Measured $r_e$')
    plt.ylabel(r'Catalog $r_e$')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()    
    
    print "Mean offset between measured and cataloged re (percentage):", np.round(100*np.mean((mdf['re_M']-mdf['re_C'])/mdf['re_C']),decimals=2)
    print "Mean abs offset between measured and cataloged re (percentage):", np.round(100*np.mean(abs(mdf['re_M']-mdf['re_C'])/mdf['re_C']),decimals=2)
    
    print "Median offset between measured and cataloged re (percentage):", np.round(100*np.median((mdf['re_M']-mdf['re_C'])/mdf['re_C']),decimals=2)
    print "Median abs offset between measured and cataloged re (percentage):", np.round(100*np.median(abs(mdf['re_M']-mdf['re_C'])/mdf['re_C']),decimals=2)
 
    print "Mean abs dex difference between measured and catalog re:", np.mean(np.sqrt((mdf['logr_M']-mdf['logr_C'])**2))
    print "Median abs dex difference between measured and catalog re:", np.median(np.sqrt((mdf['logr_M']-mdf['logr_C'])**2))
    print "Mean abs diff between measured and catalog re:", np.mean(abs(mdf['re_M']-mdf['re_C']))
    print "Median abs diff between measured and catalog re:", np.median(abs(mdf['re_M']-mdf['re_C']))
    print "Mean difference between measured and catalog re:", np.mean(mdf['re_M']-mdf['re_C'])
    print "Median difference between measured and catalog re:", np.median(mdf['re_M']-mdf['re_C'])    
        
    mdf['q_M'] = mdf['ar']
    mdf['q_C'] = mdf['q']
    mdf['logq_M'] = np.log10(mdf['q_M'])
    mdf['logq_C'] = np.log10(mdf['q_C'])
    
    plt.scatter(mdf['q_M'],mdf['q_C'],facecolors='none',s=10,edgecolors='k')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel(r'Measured $q$')
    plt.ylabel(r'Catalog $q$')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()    
    
    print "Mean dex difference between measured and catalog q:", np.mean(np.sqrt((mdf['logq_M']-mdf['logq_C'])**2))
    print "Median dex difference between measured and catalog q:", np.median(np.sqrt((mdf['logq_M']-mdf['logq_C'])**2))
    print "Mean abs dex diff between measured and catalog n:", np.mean(abs(mdf['logq_M']-mdf['logq_C']))
    print "Median abs dex diff between measured and catalog n:", np.median(abs(mdf['logq_M']-mdf['logq_C']))
    print "Mean dex difference between measured and catalog q:", np.mean(mdf['logq_M']-mdf['logq_C'])
    print "Median dex difference between measured and catalog q:", np.median(mdf['logq_M']-mdf['logq_C'])
 
    plt.scatter(mdf['Delta_re'],mdf['Delta_n'],facecolors='none',s=10,edgecolors='k')
    plt.ylabel(r'$\delta_n$')
    plt.xlabel(r'$\delta_{re}$')
    plt.ylim([-2,2])
    plt.axvline(0,color='k',linestyle='--')
    plt.axhline(0,color='k',linestyle='--')
    plt.xlim([-0.3,0.3])
    plt.show()
    
    plt.scatter(mdf['Delta_q'],mdf['Delta_n'],facecolors='none',s=10,edgecolors='k')
    plt.ylabel(r'$\delta_n$')
    plt.xlabel(r'$\delta_{q}$')
    plt.axvline(0,color='k',linestyle='--')
    plt.axhline(0,color='k',linestyle='--')
    plt.ylim([-2,2])
    plt.xlim([-0.3,0.3])
    plt.show()

    if sky:
#         print mdf['sky'].describe()
        reds = plt.get_cmap("Reds")
        seismic = plt.get_cmap("seismic")
        plt.scatter(mdf['n_M'],mdf['n_C'],c=mdf['sky'],cmap=seismic,s=50,edgecolors='k',alpha=0.7,vmin=-1,vmax=1)
        plt.plot([0,8],[0,8],'k--')
        plt.xlabel(r'Measured $n$')
        plt.ylabel(r'Catalog $n$')
        plt.xlim([0,8])
        plt.ylim([0,8])
        cbar = plt.colorbar(orientation='vertical')
        cbar.set_label(r'Measured Sky level (counts)', fontsize=14)
        plt.show()
    return