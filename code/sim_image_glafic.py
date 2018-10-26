import numpy as np
import os
from astropy.io import fits

dir_root = os.path.abspath(os.getcwd()+'/../')
dir_data = dir_root + '/data'
dir_code = dir_root + '/code'
dir_exe = dir_root + '/exe'

glafic_exe = dir_exe + '/glafic'

def PSF_command(paras_psf1, paras_psf2, psf_ratio):
    FWHM1, e1, theta_e1, beta1 = paras_psf1
    FWHM2, e2, theta_e2, beta2 = paras_psf2

    command = 'psf %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'\
        %(FWHM1, e1, theta_e1, beta1, FWHM2, e2, theta_e2, beta2, psf_ratio)
    return command


class OneObject(object):
    def __init__(self, redshift, mass_component=[], light_component=[]):
        self.Masslist = mass_component.copy()
        self.Lightlist = light_component.copy()
        self.redshift = redshift
        self.get_component_numbers()

    def get_component_numbers(self):
        self.n_mass = len(self.Masslist)
        self.n_light = len(self.Lightlist)

    def add_mass_component(self, mass_component):
        self.Masslist.append(mass_component)
        self.get_component_numbers()

    def reset_mass_component(self):
        self.Masslist = []
        self.get_component_numbers()

    def add_light_component(self, light_component):
        self.Lightlist.append(light_component)
        self.get_component_numbers()

    def reset_light_component(self):
        self.Lightlist = []
        self.get_component_numbers()


class LensSystem(object):

    def __init__(self, lensobj, sourceobj,\
                 kwargs_cosmo={'omega':0.3, 'lambda':0.7, 'hubble':0.7}):
        self.lens = lensobj
        self.source = sourceobj
        self.kwargs_cosmo = kwargs_cosmo

    def _get_primary_command(self, **kwargs):
        # Primary Parameters #
        omega_m = self.kwargs_cosmo['omega']
        omega_l = self.kwargs_cosmo['lambda']
        hubble = self.kwargs_cosmo['hubble']

        zl = self.lens.redshift

        xmin, xmax = kwargs['x_range']
        ymin, ymax = kwargs['y_range']
        prefix = kwargs['prefix']
        pix_ext = kwargs['pix_ext']
        pix_poi = kwargs['pix_poi']

        primary_param_str =\
'''
# primary parameters
omega    %.3f
lambda   %.3f
weos     -1.000
hubble   %.3f
zl       %.3f
prefix   %s
xmin     %.3f
ymin     %.3f
xmax     %.3f
ymax     %.3f
pix_ext  %.3f
pix_poi  %.3f
maxlev   6
'''%(omega_m, omega_l, hubble, zl, prefix, xmin, ymin, xmax, ymax,\
     pix_ext, pix_poi)

        return primary_param_str

    def _get_secondary_command(self, kwargs_secondary):
        command = ''
        for item in kwargs_secondary.items():
            key, value = item
            command += '%s  %s\n'%(key, value)

        return command

    def _get_lens_command(self):

        lensstr = ''

        # lens model command #
        for idx_lens in range(self.lens.n_mass):
            lens_mass = self.lens.Masslist[idx_lens]
            lens_x = lens_mass['x_center']
            lens_y = lens_mass['y_center']
            lens_ellipticity = lens_mass['ellipticity']
            lens_pa = lens_mass['pa']
            lens_type = lens_mass['Type']

            if lens_type == 'nfw':
                key_mass = lens_mass['mass']
                key_6 = lens_mass['c']
                key_7 = 0

            elif lens_type == 'sers':
                key_mass = lens_mass['mass']
                key_6 = lens_mass['re']
                key_7 = lens_mass['n']

            elif lens_type == 'pow':
                key_mass = lens_mass['mass']
                key_6 = lens_mass['r_ein']
                key_7 = lens_mass['gamma']

            elif lens_type == 'sie':
                key_mass = lens_mass['sigma']
                key_6 = lens_mass['r_core']
                key_7 = 0

            else:
                raise UnkownMassProfileError(\
                                'Unkown Lens Mass Profile %s'%lens_type)

            lensstr += 'lens %s %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'\
                    %(lens_type, key_mass, lens_x, lens_y,\
                      lens_ellipticity, lens_pa, key_6, key_7)

        return lensstr

    def _get_source_command(self, islens=False):
        # source model command #
        sourcestr = ''
        if islens:
            source_to_model = self.lens
        else:
            source_to_model = self.source

        for idx_source in range(source_to_model.n_light):
            source = source_to_model.Lightlist[idx_source]
            source_z = source_to_model.redshift
            if islens:
                source_z +=0.1

            source_flux = source['flux']
            source_x = source['x_center']
            source_y = source['y_center']
            source_type = source['Type']

            if source_type=='point':

                sourcestr += 'extend %s %.3f %.3f %.3f %.3f\
                        %.3f %.3f %.3f %.3f\n'\
                    %(source_type, source_z, source_flux, source_x, source_y,\
                     0, 0, 0, 0)

            elif source_type=='sersic':
                source_ellipticity = source['ellipticity']
                source_pa = source['pa']
                source_re = source['re']
                source_n = source['n']

                sourcestr +=\
                    'extend %s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'\
                    %(source_type, source_z, source_flux, source_x, source_y,\
                    source_ellipticity, source_pa, source_re, source_n)

            else:
                raise UnkownLightProfileError(\
                                'Unkown Source Light Profile %s'%lens_type)
        return sourcestr

    def _get_pointsource_command(self):
        pointsourcestr = ''

        source_to_model = self.source
        for idx_source in range(source_to_model.n_light):
            source = source_to_model.Lightlist[idx_source]
            source_z = source_to_model.redshift
            source
            source_x = source['x_center']
            source_y = source['y_center']

            pointsourcestr += 'point %.3f %.3f %.3f\n'\
                    %(source_z, source_x, source_y)

        return pointsourcestr

    def _get_object_command(self, psfinfo, islens=False, ispointsource=False):
        psf1, psf2, psfratio = psfinfo

        num_lens = self.lens.n_mass
        if ispointsource:
            num_ps = self.source.n_light
            num_ext = 0
        else:
            num_ext = self.source.n_light
            num_ps = 0

        object_str = 'startup %d %d %d\n'%(num_lens, num_ext, num_ps)

        lensstr = self._get_lens_command()
        if ispointsource:
            sourcestr = self._get_pointsource_command()
            psfstr = ''
        else:
            sourcestr = self._get_source_command(islens=islens)
            psfstr = PSF_command(psf1, psf2, psfratio)

        object_str = object_str + lensstr + sourcestr + psfstr\
                + 'end_startup\n'

        return object_str

    def _get_optimization_command(self, command=''):
        if len(command)==0:
            command = ''
        else:
            command = 'start_setopt\n'+command+'\nend_setopt\n'
        return command

    def _get_operation_command(self, operation_list):
        operation_command = 'start_command\n'
        for operation in operation_list:
            operation_command += '%s\n'%(operation)

        operation_command += 'quit\n'
        return operation_command

    def glafic_command(self, prefix, x_range, y_range,\
                       psfinfo, pix_ext, pix_poi, operation_list,\
                       kwargs_secondary, islens=False,\
                       ispointsource=False):

        primary_command = self._get_primary_command(\
                    pix_ext=pix_ext, prefix=prefix, pix_poi=pix_poi,\
                    x_range=x_range, y_range=y_range)
        secondary_command = self._get_secondary_command(kwargs_secondary)
        object_command = self._get_object_command(psfinfo, islens=islens,\
                                                  ispointsource=ispointsource)
        optimization_command = self._get_optimization_command()
        operation_command = self._get_operation_command(operation_list)

        allcommand = primary_command + secondary_command\
                + object_command +optimization_command + operation_command

        return allcommand

    def makelens(self, prefix, x_range, y_range, psfinfo, pix_ext, pix_poi,\
                 command_list=['writeimage'], ispointsource=False,\
                 islens=False, background=0, noise=0, **kwargs):

        tmp_input_file = 'tmp%s%d.input'%(prefix, np.random.randint(100))
        allcommand = self.glafic_command(prefix, x_range, y_range, psfinfo,\
                                pix_ext, pix_poi, command_list, kwargs,\
                                ispointsource=ispointsource, islens=islens)
        f = open(dir_exe + '/' + tmp_input_file, 'w+')
        f.write(allcommand)
        f.close()

        os.system(glafic_exe + ' %s/%s'%(dir_exe, tmp_input_file))
        os.system('rm %s/%s'%(dir_exe, tmp_input_file))

    def sim_image(self, prefix, x_range, y_range, psfinfo, pix_ext, pix_poi,\
                  output, newrun_source=True, newrun_lens=True,\
                  background=0, noise=0, clean=True, **kwargs):

        # Don't mind the confusing filenames... #
        lensed_source_image = dir_code + '/' + prefix + '_image.fits'
        lens_image = dir_code + '/' + prefix + '_source.fits'

        command_suffix_source = ' %f %f'%(background, noise)
        command_suffix_lens = ' %f %f'%(0, noise)

        # avoid repeating work #
        if (not os.path.exists(lensed_source_image)) or newrun_source:
            print('Generating source fluxes')
            self.makelens(prefix, x_range, y_range, psfinfo,\
                          pix_ext, pix_poi,\
                          command_list=['writeimage'+command_suffix_source],\
                          islens=False, **kwargs)
        if (not os.path.exists(lens_image)) or newrun_lens:
            print('Generating lens fluxes')
            self.makelens(prefix, x_range, y_range, psfinfo,\
                          pix_ext, pix_poi,\
                          command_list=['writeimage_ori'+command_suffix_lens],\
                          islens=True, **kwargs)

        header = fits.open(lensed_source_image)[0].header
        sourceimg = fits.open(lensed_source_image)[0].data
        lensimg = fits.open(lens_image)[0].data

        image = sourceimg + lensimg

        hdusource = fits.ImageHDU(data=sourceimg, name='SOURCE')
        hdulens = fits.ImageHDU(data=lensimg, name='LENS')
        hduall = fits.ImageHDU(data=image, name='ALL')
        hdu0 = fits.PrimaryHDU(header=header)

        hdulist = fits.HDUList([hdu0, hdusource, hdulens, hduall])
        hdulist.writeto(output, overwrite=True)

        if clean:
            os.system('rm ' + dir_code + '/' + prefix + '_image.fits')
            os.system('rm ' + dir_code + '/' + prefix + '_source.fits')

        return output

    def findimage(self, prefix, x_range=[-10, 10], y_range=[-10, 10],\
                  psfinfo=[None, None, None], pix_ext=0.2, pix_poi=3, **kwargs):

        self.makelens(prefix, x_range, y_range, psfinfo, pix_ext, pix_poi,\
                      command_list=['findimg'], ispointsource=True, **kwargs)
        return prefix + '_point.dat'

    def writecrit(self, prefix, x_range=[-10, 10], y_range=[-10, 10],\
                  psfinfo=[None, None, None], pix_ext=0.2, pix_poi=3, **kwargs):
        self.makelens(prefix, x_range, y_range, psfinfo, pix_ext, pix_poi,\
                      command_list=['writecrit'], ispointsource=True, **kwargs)
        return prefix + '_crit.dat'


def main():

    lens_nfw = {'Type':'nfw', 'mass':1e13, 'c':10,\
                'ellipticity':0, 'pa':0, 'x_center':0, 'y_center':0}
    lens_serc = {'Type':'sers', 'mass':2e11, 're': 0.7, 'n':1,\
                'ellipticity':0.3, 'pa':30, 'x_center':0, 'y_center':0}

    lens_sie = {'Type':'sie', 'sigma':200, 'r_core':0.01,\
               'ellipticity':0.3, 'pa':30, 'x_center':0, 'y_center':0}

    lens_light = {'Type':'sersic', 'flux':10, 'x_center':0., 'y_center':0.,\
                  'ellipticity':0.3, 'pa':30, 're':0.7, 'n':1}
    source_light = {'Type':'point', 'flux':1, 'x_center':0.5, 'y_center':0.4}

    do_something = 1
    lens = OneObject(0.5, mass_component=[lens_sie],\
               light_component=[lens_light])
    source = OneObject(5, light_component = [source_light])

    lenssystem = LensSystem(lens, source)

    prefix = 'out'

    psf1 = [0.8, 0.02, 60.0, 5.0]
    psf2 = [1.2, 0.02, -30.0, 3.0]
    psfratio = 1.0
    psfinfo = [psf1, psf2, psfratio]

    x_range = (-10, 10)
    y_range = (-10, 10)

    minrange = np.min([x_range[1]-x_range[0], y_range[1]-y_range[0]])

    pix_ext = 0.2
    pix_poi = 1.0

#    lenssystem.sim_image(prefix, x_range, y_range, psfinfo, pix_ext, pix_poi,\
#                         output='test.fits', newrun_lens=True,\
#                        psfconv_size=minrange/5, seeing_sub=5, flag_extnorm=1)

    lenssystem.writecrit('out')

if __name__=='__main__':
    main()
