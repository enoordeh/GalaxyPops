from __future__ import division
import sys, os, glob, fnmatch
sys.path.insert(0,'/data/emiln/XLSSC122_GalPops/Analysis/Modules')
from GalfitPyWrap import galfitwrap as gf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from astropy.table import Table, hstack, join
import multiprocessing as mp
import logging as log

def load_HST_galaxy_catalog(filename):
    df = pd.read_csv(filename,delim_whitespace=True)
    df['gold_cluster'] = 0
    df['gold_foreground'] = 0
    df['silver_cluster'] = 0
    # df.columns.values
    # print df
    df['(f140)kron'] = df['f140_kron']
    for i,r in df.iterrows():
        if (r['member1']>0.5) and (r['(f140)kron']<24 or (r['(f140)kron']<24.5 and r['em-code']==1)):
            df.at[i,'gold_cluster'] = 1
        if (r['member1']<0.5 and r['member1']>0.1) and (r['(f140)kron']<24 or (r['(f140)kron']<24.5 and r['em-code']==1)):
            df.at[i,'silver_cluster'] = 1
        if (r['member2']>0.5) and (r['(f140)kron']<24 or (r['(f140)kron']<24.5 and r['em-code']==1)):
            df.at[i,'gold_foreground'] = 1  

    df['color'] = df['f105_0p8'] - df['f140_0p8']
    df['F140W'] = df['f140_kron']
    df['ID'] = df['phot-id']
    df['RA'] = df['ra']
    df['DEC'] = df['dec']

    # Compute r/r500 for each cluster member
    # Cluster Center
    cluster_ra = 34.434125
    cluster_dec = -3.7587388
    r500 = 35 # arcsec
    df['r_center'] = np.sqrt((df['ra']-cluster_ra)**2 + (df['dec']-cluster_dec)**2)*3600./r500

    f140_file = '/data/emiln/XLSSC122_GalPops/Data/HST/Raw/xlssuj0217-0345-f140w_drz_sci.fits'
    hdulist = fits.open(f140_file)
    w = wcs.WCS(hdulist[0].header)
    # df = pd.read_csv(cat_file,delim_whitespace=True)
    df['X'] = w.wcs_world2pix(df['ra'],df['dec'], 1)[0]
    df['Y'] = w.wcs_world2pix(df['ra'],df['dec'], 1)[1]

    members1 = df[df['gold_cluster']==1]
    members2 = df[df['gold_foreground']==1]
    members1s = df[df['silver_cluster']==1]

    members = df[(df['gold_cluster']==1) | (df['silver_cluster']==1)]

    return df, members

def load_3DHST_galaxy_catalog(filename, mag='F140W', magthresh=24, z=2.00,
    z_thresh=None, goodfit=True, overwrite=False, verbose=True):
    '''
    Load 3DHST galaxy catalog into a Pandas DataFrame formatted for use with Galfit.

   Args:
        filename (str): File location of csv catalog 
        magthresh (int): A magnitude threshold to filter catalog by (default is 24)
        verbose (bool): Print logging info (default is True)
        z (float): Central redshift to select (default is 1.98)
        z_thresh (float): Width of selection around z (default is None)
        goodfit (bool): Only allow objects with cataloged good fits (default is True)
        overwrite (bool): Overwrite saved version of catalog (default is False)

    Returns:
        df - Catalog formatted as a Pandas DataFrame formatted for use with Galfit
    '''

    if verbose:
        def vprint(*args):
            for arg in args:
                print arg,
            print
    else:   
        vprint = lambda *a: None      # do-nothing function

    save_root = '/data/emiln/XLSSC122_GalPops/Data/3DHST/Products/catalogs/'
    if z_thresh:
        save_name = '{}magthresh{}_z{}_zthresh{}.csv'.format(save_root,magthresh,z,z_thresh)
    elif magthresh and not z_thresh:
        save_name = '{}magthresh{}.csv'.format(save_root,magthresh)
    else:
        save_name = '{}nothresh.csv'.format(save_root)

    vprint('Checking if catalog exists at {}'.format(save_name))
    if os.path.exists(save_name) and (not overwrite):
        vprint('Catalog already exists, returning DataFrame')
        return pd.read_csv(save_name)
    else:
        vprint('Catalog does not exist')
        vprint('Creating catalog...')
        df = pd.read_csv(filename,header=0,delim_whitespace=True,skiprows=[1,2,3])
        df['F140W'] = 25.0 - 2.5 * np.log10(df['f_F140W'])
        df['F140W_fixed'] = df['F140W']-1.465
        df['X'] = df['x']
        df['Y'] = df['y']
        df['ID'] = df['id']
        df['RA'] = df['ra']
        df['DEC'] = df['dec']
        if magthresh:
            vprint("Filtering objects to {}<{}".format(mag,magthresh))
            df = df[(df[mag]<magthresh) & (df[mag]>15)]
        if z_thresh:
            vprint("Filtering objects to {}<z<{}".format(z-z_thresh,z+z_thresh))
            z_df = pd.read_csv(save_root+'cosmos_3dhst.v4.1.cats/Catalog/cosmos_3dhst_v4.1.5_catalogs/cosmos_3dhst.v4.1.5.zbest.fout',delim_whitespace=True)
            df = df.merge(z_df,on='id',how='inner')
            df = df[abs(df['z']-z) < z_thresh]
        if goodfit:
            vprint("Filtering to objects with good catalog fits")
            df = df[df['use_phot']==1]
            df = df[df['flags']<2] # Flag for good fit

        vprint('Saving catalog to {}'.format(save_name))
        df.to_csv(save_name,index=False)
        vprint('Returning DataFrame')

    return df

def load_COSMOS_galaxy_catalog(filename, mag='F125W_Kron', magthresh=24, z=1.98,
    z_thresh=None, goodfit=True, overwrite=False, verbose=True):
    '''
    Load COSMOS galaxy catalog into a Pandas DataFrame formatted for use with Galfit.

    Args:
        filename (str): File location of csv catalog 
        magthresh (int): A magnitude threshold to filter catalog by (default is 24)
        verbose (bool): Print logging info (default is True)
        z (float): Central redshift to select (default is 1.98)
        z_thresh (float): Width of selection around z (default is None)
        goodfit (bool): Only allow objects with cataloged good fits (default is True)
        overwrite (bool): Overwrite saved version of catalog (default is False)

    Returns:
        df - Catalog formatted as a Pandas DataFrame formatted for use with Galfit
    '''

    if verbose:
        def vprint(*args):
            for arg in args:
                print arg,
            print
    else:   
        vprint = lambda *a: None      # do-nothing function

    save_root = '/data/emiln/XLSSC122_GalPops/Data/COSMOS/Products/catalogs/'
    if z_thresh:
        save_name = '{}magthresh{}_z{}_zthresh{}.csv'.format(save_root,magthresh,z,z_thresh)
    else:
        save_name = '{}magthresh{}.csv'.format(save_root,magthresh)

    vprint('Checking if catalog exists at {}'.format(save_name))
    if os.path.exists(save_name) and (not overwrite):
        vprint('Catalog already exists, returning DataFrame')
        return pd.read_csv(save_name)
    else:
        vprint('Catalog does not exist')
        vprint('Creating catalog...')
        df = pd.read_csv(filename)
        vprint("Filtering objects to {}<{}".format(mag,magthresh))
        # df = df[df['mag']<magthresh] # 'mag' is vanderwels catalog mag
        df = df[df[mag]<magthresh]
        if z_thresh:
            vprint("Filtering objects to {}<z<{}".format(z-z_thresh,z+z_thresh))
            df = df[abs(df['z_best']-z) < z_thresh]
        if goodfit:
            vprint("Filtering to objects with good catalog fits")
            df = df[(df['dn']/df['n'])<0.2]
            df = df[df['f']==0] # Flag for good fit

        cos_data_root = '/data/emiln/XLSSC122_GalPops/Data/Raw/COSMOS/'
        cos_file = cos_data_root+'hlsp_candels_hst_wfc3_cos-tot_f125w_v1.0_drz.fits'

        vprint('Determining pixel coords of sources using WCS info from {}'.format(cos_file))
        cos_hdulist = fits.open(cos_file)
        full_wcs = wcs.WCS(cos_hdulist[0].header)
        df['X'] = full_wcs.wcs_world2pix(df['RA'],df['DEC'], 1)[0]
        df['Y'] = full_wcs.wcs_world2pix(df['RA'],df['DEC'], 1)[1]
        vprint('Saving catalog to {}'.format(save_name))
        df.to_csv(save_name,index=False)
        vprint('Returning DataFrame')
    return df


def run_galfit_parallel(params,survey='COSMOS',fit_df=None,full_df=None,ZP=26,
    width=90,HLRwidth=False,PSFf=1,
    usePSF=True,timeout=300,verbose=False,PSF_file=None,
    sigma_file=None,data_file=None,
    save_name=None,
    PA_INIT=45, AR_INIT=0.5, MAG_INIT=21,useDYNMAG=True, convbox='50 50',
    constraint_file='none',image_width = 200, badmask='none',
    fitMagnitude=True, neighbourMagThresh=5,
    sky='Default',sky_INIT=0, N=0, df_name='',
    **kwargs):
    '''
    params = {
        'survey':survey, # {'COSMOS','3DHST','HST'} default is 'COSMOS'
        'fit_df':fit_df, # Dataframe with objects to be fit
        'full_df':full_df, # Unfiltered source catalog used for fitting neighbours
        'width':width, # Fitting region width in pixels
        'HLRwidth':HLRwidth, # Fitting region width in # of HLR
        'sigma_file':sigma_file, # Filename of sigma maps for sources
        'data_file':data_file, # Filename for data cutouts for each source
        'PSF_file':psf_file, # File_name of PSF to be used
        'usePSF':True, # Use PSF in fitting?
        'timeout':timeout, # Max runtime per object in seconds
        'PSFf':PSFf, # Fine sampling factor
        'verbose':verbose, # Verbose mode
        'PA_INIT':PA_INIT, # Initial position angle
        'AR_INIT':AR_INIT, # Initial axis ratio
        'MAG_INIT':MAG_INIT, # Initial magnitude
        'convbox':convbox, # Region to convolve PSF with in pixels (i.e. '100 100')
        'ZP':ZP, # Zeropoint 
        'constraint_file':constraint_file, # Galfit constraint filename
        'image_width':image_width, # Size of data+sigma images being used (200 for COSMOS cutouts)
        'useDYNMAG':DYNMAG, # Initialize magnitudes from catalog?
        'badmask':badmask, # filename for bad pixel mask
        'fitMagnitude':fitMagnitude, # Fit magnitudes?
        'sky':sky, # Sky fitting mode for galfit (i.e. 'default')
        'sky_INIT':sky_INIT, # Initial sky level
        'neighbourMagThresh':neighbourMagThresh, # Additional magnitude threshhold to fit neighbours (i.e. 3 -> only neighbours with mag < source_mag+3 are fit)
        'df_name': df_name, # Descriptive name of catalog being fit
        'save_name':save_name # Filename to save results to, overrides default
    }
    '''
    
    pool = mp.Pool(4) 
    PSF_name = PSF_file.split('/')[-1].split('.')[0]

    if save_name == None:
        save_name = df_name+'_'+data_file[1:-5]+'_'+sigma_file[1:-5]+'_w'+\
                    [str(HLRwidth)+'HLR' if HLRwidth!=False else str(width)][0]+'_'+PSF_name+'_'+\
                    str(int(timeout/60))+'min_CONV'+convbox.split(' ')[0]+\
                    ['_CONSTR' if constraint_file !='none' else ''][0]+['_DYNMAG' if useDYNMAG else ''][0]
        params['save_name'] = save_name
    print('SAVE_NAME:{}'.format(save_name))

    results = [pool.apply_async(run_galfit, args=([r]), kwds=params) for _, r in fit_df.iterrows()]
    output = [p.get() for p in results]
    pool.close()

    new_df = pd.DataFrame(output,columns=['ID','ra','dec','re','ar','n','mag','sky','chi2nu','ErrorValue']) 
    # ErrorValue of 124 = process timeout, 1 = GALFIT exception
    # results_root = '/data/emiln/XLSSC122_GalPops/Analysis/COSMOS/Results/'
    results_root = '/data/emiln/XLSSC122_GalPops/Analysis/'+survey+'/Results/'
    save_file = results_root+save_name+'.csv'
    print('Saving results to {}'.format(save_file))
    new_df.to_csv(save_file,index=False)
    print('Returning results as DataFrame')
    return new_df, save_name


def run_galfit(row,survey='COSMOS',full_df=None,ZP=26,width=90,HLRwidth=False,PSFf=1,
    usePSF=True,timeout=300,verbose=False,PSF_file='tinytim_psf.fits',
    sigma_file='/sigma_meanexp_cutout.fits',data_file='/cutout.fits',
    save_name='rmssigmameanexp_w180_TESTING',
    PA_INIT=45, AR_INIT=0.5, MAG_INIT=21,useDYNMAG=True, convbox='100 100',
    constraint_file='none',image_width = 200, badmask='none',
    fitMagnitude=True, neighbourMagThresh=5,
    sky='Default',sky_INIT=0, N=0, **kwargs):
#     cutout_width = 200 # ALWAYS KEEP THE SAME, PHYSICAL SIZE OF CUTOUTS FROM DATA_PREP
#     N = 0 # Extra width to search for neighbours in?
#     mag_thresh = 4 # Neighbours only fit if less than mag_thresh fainter than primary
    
    df = full_df
    r = row
    ra = r.RA
    dec = r.DEC
    print("***{}***".format(survey))
    if survey == 'COSMOS': IDname = 'NUMBER'
    if survey == '3DHST': IDname = 'ID'
    if survey == 'HST': IDname = 'ID' 
    ID = int(r[IDname])
    # tdir = '/data/emiln/XLSSC122_GalPops/Data/Products/COSMOS/galfit_results/'+str(ID)
    tdir = '/data/emiln/XLSSC122_GalPops/Data/Products/'+survey+'/galfit_results/'+str(ID)
    
    if badmask != 'none':
        print "Using bad pixel mask..."
        badmask = tdir+badmask
    if useDYNMAG:
        if survey == 'HST':
            MAG_INIT=np.round(r.F140W,decimals=2)
            og_mag=MAG_INIT
            print "Initializing",str(ID),"with F140W Kron magnitude:", MAG_INIT
        elif survey == '3DHST':
            MAG_INIT=np.round(r.F140W,decimals=2)
            og_mag=MAG_INIT
            print "Initializing",str(ID),"with F140W Kron magnitude:", MAG_INIT
        else:
            MAG_INIT=np.round(r.F125W_Kron,decimals=2)
            og_mag=MAG_INIT
            print "Initializing",str(ID),"with F125W Kron magnitude:", MAG_INIT
    # pixcrd = full_wcs.wcs_world2pix(ra,dec, 1)
    # print pixcrd
    # X = int(pixcrd[0])
    # Y = int(pixcrd[1])
    X,Y = r.X,r.Y
    
    CX, CY = image_width, image_width
    
    if fitMagnitude:
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
    print "Cutout width:", image_width


    
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

    if HLRwidth == False:
        bounds = [CX-width,CX+width,CY-width,CY+width]
        bounds2 = [X-width,X+width,Y-width,Y+width]
        print "Cutoutwidth (pixels):", width*2
        print "Cutoutwidth (arcsec):", width*2*0.06
    else: 
        if survey == 'COSMOS': width = int(np.ceil(HLRwidth*r.HLR))
        if survey == '3DHST': width = int(np.ceil(HLRwidth*r.flux_radius))
        if survey == 'HST': width = int(np.ceil(HLRwidth*r.asec_fwhm/0.06))
        bounds = [CX-width,CX+width,CY-width,CY+width]
        bounds2 = [X-width,X+width,Y-width,Y+width]
        print "Cutoutwidth (pixels) for ID",str(ID),":", width*2
        print "Cutoutwidth (arcsec) for ID",str(ID),":", width*2*0.06
        
    print "Bounds:", bounds

    # Find neighbours in full DF
    # Neighbour check needs to be done on original fits image / catalog
    # ndf = df[(df['X']>(bounds2[0]-N)) & (df['X']<(bounds2[1]+N)) & (df['Y']>(bounds2[2]-N)) & (df['Y']<(bounds2[3]+N))
    #          & (df['NUMBER']!=ID)]
    ndf = df[(df['X']>(bounds2[0]-N)) & (df['X']<(bounds2[1]+N)) & (df['Y']>(bounds2[2]-N)) & (df['Y']<(bounds2[3]+N))
             & (df[IDname]!=ID)]

    print len(ndf),"NEIGHBOURS FOUND"

    print "Adding additional model components for neighbours..."
    for row in ndf.iterrows():
        r = row[1]
        NX = r.X - X + image_width
        NY = r.Y - Y + image_width
        if useDYNMAG:
            if survey == '3DHST': MAG_INIT=np.round(r.F140W,decimals=2)
            if survey == 'HST': MAG_INIT=np.round(r.F140W,decimals=2)
            if survey == 'COSMOS': MAG_INIT=np.round(r.F125W_Kron,decimals=2)
            if np.isnan(MAG_INIT):
                MAG_INIT=og_mag # Initialize with MAG of primary target
                print "***NEIGHBOUR MAG NOT CATALOGED***"
            print "NEIGHBOUR initialized with magnitude:", MAG_INIT
            if MAG_INIT > og_mag+neighbourMagThresh:
                print "Neighbour mag too faint, not being fit"
                continue # If mag is too faint, do not fit this object 
                
        # seqnr = int(r['NUMBER'])
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


    if usePSF:
        print "Using PSF"
        O=gf.CreateFile(tdir+data_file, bounds, model,fout=tdir+'/input.feedme',\
                        Pimg=PSF_file, Simg=tdir+sigma_file, Oimg=tdir+'/out.fits', ZP=ZP, scale='0.06 0.06',PSFf=PSFf, convbox=convbox,
                       constr=constraint_file, badmask=badmask, sky=sky,skyINIT=sky_INIT)
        # convbox should be larger than PSF. f125_400 psf is (166,166)
    # else:
    #     O=gf.CreateFile(tdir+data_file, bounds, model,fout=tdir+'/input.feedme',\
    #                     Simg=tdir+sigma_file, ZP=zp, scale='0.06 0.06', PSFf=PSFf, convbox=convbox, constr=constraint_file,
    #                    badmask=badmask,sky=sky,skyINIT=sky_INIT)
    pout,oimg,mods,EV,chi2nu=gf.rungalfit(tdir+'/input.feedme',verb=verbose, outfile=tdir+'/out.fits',timeout=timeout,cwd=tdir)
    # if verbose:
    #     for l in pout:
    #         print l
    if EV==124: print "***PROCESS TIMEOUT***"
    print '****',ID,'****'
    try: 
        # os.rename('out.fits',tdir+'/out_'+save_name+'.fits')
        # os.rename('fit.log',tdir+'/fit_'+save_name+'.log')
        # os.rename('galfit.01',tdir+'/galfit_'+save_name+'.01')
        os.rename(tdir+'/out.fits',tdir+'/out_'+save_name+'.fits')
        os.rename(tdir+'/fit.log',tdir+'/fit_'+save_name+'.log')
        os.rename(tdir+'/galfit.01',tdir+'/galfit_'+save_name+'.params')
    except:
        print "***",ID,"Fit failed at GALFIT level***"
        # return pout
        

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
            EV+=2**2
            print 'ID',str(ID),'len(re_val)>1', len(re_val)>1

        if len(ar_val)>1:
            EV+=2**3
            print 'ID',str(ID),'len(ar_val)>1', len(ar_val)>1

        if len(n_val)>1:
            EV+=2**4
            print 'ID',str(ID),'len(n_val)>1', len(n_val)>1

        if len(m_val)>1:
            EV+=2**5
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

def plot_by_ID(ID,save_name=None,survey='COSMOS'):
    if not save_name:
        print 'provide a save_name...'
        return
    tdir = '/data/emiln/XLSSC122_GalPops/Data/Products/'+survey+'/galfit_results/'+str(ID)
    
    results_root = '/data/emiln/XLSSC122_GalPops/Analysis/'+survey+'/Results/'
    ttdf = pd.read_csv(results_root+save_name+'.csv')
    print ttdf[ttdf['ID'] == ID]
    try: 
        oimg = fits.open(tdir+'/out_'+save_name+'.fits')
        showme3(oimg)
    except:
        print "*** No model fit found. Plotting raw image from /cutout.fits ***"
        try:
            oimg = fits.open(tdir+'/cutout.fits')
            vm = np.percentile(oimg[0].data,99)
            plt.figure()
            plt.subplot(131)
            plt.imshow(oimg[0].data,interpolation='none',cmap='viridis',vmin=1e-4,vmax=vm)
        except:
            print "*** No cutout.fits file found. Plotting CPS file. ***"
            oimg = fits.open(tdir+'/data_cps.fits')
            vm = np.percentile(oimg[0].data,99)
            plt.figure(figsize=(5,5))
            plt.subplot()
            plt.imshow(oimg[0].data,interpolation='none',cmap='viridis',vmin=1e-4,vmax=vm)
        
    # print '\n ***CATALOG PARAMS***'
    # print cos_df[cos_df['NUMBER']==ID][['q','n','re']]
    plt.show()
    return

def load_COSMOS_published_cat():
    cosmos_cat_root = '/data/emiln/XLSSC122_GalPops/Data/Products/COSMOS/catalogs/'
    cos_cat = cosmos_cat_root+'cos_2epoch_wfc3_f125w_060mas_v1.0_galfit.cat'
    cos_df = pd.read_csv(cos_cat,header=0,delim_whitespace=True,skiprows=[1])
    cos_df2 = cos_df[cos_df.columns[:-1]]
    cos_df2.columns = cos_df.columns[1:]
    return cos_df2

def compare_COSMOS(save_name,catalog_df,sky_test=False):
    # Load fits and compare to published catalog
    results_root = '/data/emiln/XLSSC122_GalPops/Analysis/COSMOS/Results/'
    df2 = pd.read_csv(results_root+save_name+'.csv')
    # df2 = pd.read_csv(galfit_file)
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
    
    # print mdf.columns.values
    
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

    if sky_test:
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