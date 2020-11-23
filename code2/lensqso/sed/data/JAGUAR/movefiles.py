import os, sys

def clean_dir(direct):
    info_dir = direct + '/info'
    spec_dir = direct + '/spec'
    os.system('mkdir %s'%info_dir)
    os.system('mkdir %s'%spec_dir)

    # info files

    rlz = os.path.basename(direct)
    addpath = '/neogal.iap.fr/JAGUAR_mock_catalogue/%s'%rlz

    info_Q = direct + '/' + addpath + '/JADES_Q_mock_%s_v1.2.fits.gz'%rlz
    info_SF = direct + '/' + addpath + '/JADES_SF_mock_%s_v1.2.fits.gz'%rlz

    spec_Q = direct + '/' + addpath + '/JADES_Q_mock_%s_v1.2_spec_5A_30um.fits.gz'%rlz
    spec_SF_0 = direct + '/' + addpath + '/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_0p2_1.fits'%rlz
    spec_SF_1 = direct + '/' + addpath + '/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_1_1p5.fits'%rlz
    spec_SF_2 = direct + '/' + addpath + '/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_1p5_2.fits'%rlz

    os.system('mv %s %s'%(info_Q, info_dir))
    os.system('mv %s %s'%(info_SF, info_dir))
    os.system('mv %s %s'%(spec_Q, spec_dir))
    os.system('mv %s %s'%(spec_SF_0, spec_dir))
    os.system('mv %s %s'%(spec_SF_1, spec_dir))
    os.system('mv %s %s'%(spec_SF_2, spec_dir))

def main():
    clean_dir('/Users/minghao/Research/Projects/lensQSO/code2/lensqso/sed/data/r11')

if __name__=='__main__':
    main()
