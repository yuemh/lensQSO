import os, sys
import numpy as np
from astropy.table import Table, vstack

def WISE_download_command(input_file, output_file, catalog='cwcat', radius=2):
    command = 'curl -F filename=@%s'%(input_file)\
            + ' -F catalog=%s'%(catalog)\
            + ' -F spatial=upload'\
            + ' -F uradius=%.2f'%(radius)\
            + ' -F outfmt=1'\
            + ' -F selcols="source_name,source_id,ra,dec,w1mpro,w1sigmpro,w2mpro,w2sigmpro"'\
            + ' "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"'\
            + ' -o %s'%(output_file)\

    return command

def script_CatWISE(tbl, ra_col='raStack', dec_col='decStack', n_onematch=200000):
#    ntask = np.random.randint(0, 1000)
    ntask = 3

    script = open('download_script.txt', 'w')

    Niter = int(len(tbl)/n_onematch)

    for niter in range(Niter+1):
        start = niter * n_onematch
        end = np.min([(niter+1)*n_onematch, len(tbl)])
        print('Generating coordinate list for objects %d to %d'%(start, end-1))
        coord_tbl = tbl[ra_col, dec_col][start:end]
        coord_tbl.rename_column(ra_col, 'ra')
        coord_tbl.rename_column(dec_col, 'dec')
        coord_tbl.write('coord%d_%d.tbl'%(niter, ntask),\
                        format='ascii.ipac', overwrite=True)

        input_file = 'coord%d_%d.tbl'%(niter, ntask)
        output_file = './combine/coord%d_%d_matched.tbl'%(niter, ntask)

        cmd = WISE_download_command(input_file, output_file)
        script.write(cmd)
        script.write('\n')

    script.close()

def combine_result(ntask, output):
    alltbl = []
    for root, namedir, names in os.walk('./combine'):
        for name in names:
            if '%d_matched.tbl'%(ntask) in name:
                thistbl = Table.read(root + '/' + name, format='ascii.ipac')
                alltbl.append(thistbl)
    alltbl = vstack(alltbl)
    alltbl.write(output)

def main():
    tbl_to_match = Table.read('../PanStarrs_ra_22_6.fits')
    print(len(tbl_to_match))
    script_CatWISE(tbl_to_match)
    os.system('./download_script.txt')
#    combine_result(1, 'PS_WISE_2.fits')

if __name__=='__main__':
    main()
