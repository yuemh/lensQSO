import numpy as np
from astropy.io import fits
import mylib as ml
import os
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.SimulationAPI.simulations import Simulation
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver\
        import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import lenstronomy.Plots.output_plots as lens_plot


default_cosmo = FlatLambdaCDM(H0=70, Om0=30)


class MassComponent(object):
    def __init__(**kwargs):
        if kwargs['profile'] == 'NFW':
            self.redshift = kwargs['redshift']
            self.mass = kwargs['mass']


class oneobj(object):
    def __init__(self, redshift, mass_component=[], light_component=[]):
        self.Masslist = mass_component.copy()
        self.Lightlist = light_component.copy()
        self.redshift = redshift
        self.n_mass = len(mass_component)
        self.n_light = len(light_component)

    def add_mass_component(self, mass_component):
        self.Masslist.append(mass_component)
        self.n_mass = self.n_mass + 1

    def add_light_component(self, light_component):
        self.Lightlist.append(light_component)
        self.n_light = self.n_light + 1


class LensQSO(object):

    def __init__(self, lens, source, cosmo=default_cosmo):
        self.lens = lens
        self.source = source
        self.cosmo = cosmo

        self.update_lensmodel()

    def update_lensmodel(self):
        self.lenscosmo = LensCosmo(\
                self.lens.redshift, self.source.redshift, self.cosmo)

        self.D_l = self.lenscosmo.D_d * u.Mpc
        self.D_s = self.lenscosmo.D_s * u.Mpc
        self.D_ls = self.lenscosmo.D_ds * u.Mpc
        self.crit_density = const.c**2 / 4 / np.pi / const.G \
                * self.D_s /self.D_l / self.D_ls

        self._update_lens()
        self.lensmodel = LensModel(lens_model_list = self._lens_model_list,\
                                  z_source = self.source.redshift,\
                                  z_lens = self.lens.redshift,\
                                  cosmo = self.cosmo)
        self._update_flux()

    def _update_lens(self):
        # convert the lens properties to angular space
        self._lens_model_list = []
        self._kwargs_lens = []

        for idx_lens in range(self.lens.n_mass):
            mass_component = self.lens.Masslist[idx_lens]

            if mass_component['Profile']=='NFW':
                nfw_mass = 10**mass_component['Mass']
                nfw_c = mass_component['c']
                nfw_x = mass_component['x_center']
                nfw_y = mass_component['y_center']
                Rs_angle, theta_Rs =\
                        self.lenscosmo.nfw_physical2angle(nfw_mass, nfw_c)
                self._lens_model_list.append('NFW')
                self._kwargs_lens.append(\
                        {'Rs': Rs_angle, 'theta_Rs': theta_Rs,\
                         'center_x': nfw_x, 'center_y': nfw_y})

            elif mass_component['Profile']=='Sersic':
                sersic_mass = 10**mass_component['Mass']
                incomplete = True

            elif mass_component['Profile']=='POINT_MASS':
                point_mass = 10**mass_component['Mass']
                point_x = mass_component['x_center']
                point_y = mass_component['y_center']

                theta_E = np.sqrt(float(point_mass * const.M_sun\
                            / np.pi / self.D_l**2 / self.crit_density))\
                            * 206265

                self._lens_model_list.append('POINT_MASS')
                self._kwargs_lens.append({'theta_E': theta_E,\
                        'center_x':point_x, 'center_y': point_y})

            elif mass_component['Profile']=='SPEP':
                spep_mass = 10**mass_component['Mass']
                spep_e1 = mass_component['e1']
                spep_e2 = mass_component['e2']
                spep_x = mass_component['x_center']
                spep_y = mass_component['y_center']


    def _update_flux(self):
        self._flux_lens_list = []
        self._flux_source_list = []
        self._flux_source_ps_list = []

        self._kwargs_lensflux = []
        self._kwargs_sourceflux = []
        self._kwargs_psflux = []

        for idx_lens in range(self.lens.n_light):
            light_component = self.lens.Lightlist[idx_lens]

            if light_component['Profile']=='SERSIC':
                self._flux_lens_list.append('SERSIC')

                sersic_flux = light_component['Flux']
                sersic_n = light_component['n']
                sersic_re = light_component['Re']
                sersic_x = light_component['x_center']
                sersic_y = light_component['y_center']

                self._kwargs_lensflux.append({'amp': sersic_flux,\
                    'R_sersic': sersic_re, 'n_sersic': sersic_n,\
                    'center_x': sersic_x, 'center_y': sersic_y})

        self.lightmodel_lens = LightModel(\
                                        light_model_list=self._flux_lens_list)

        for idx_source in range(self.source.n_light):
            light_component = self.source.Lightlist[idx_source]

            if light_component['Profile']=='PointSource':
                self._flux_source_ps_list.append('SOURCE_POSITION')
                ps_x = light_component['x_center']
                ps_y = light_component['y_center']
                ps_flux = light_component['Flux']

                self.pointSource = PointSource(\
                            point_source_type_list=self._flux_source_ps_list,\
                            lensModel=self.lensmodel,\
                            fixed_magnification_list=[True])

                self._kwargs_psflux.append({'ra_source': ps_x,\
                                'dec_source': ps_y, 'source_amp': ps_flux})
#                ps_x_img, ps_y_img = pointSource.image_position(\
#                                kwargs_ps=self._kwargs_sourcefluxs,\
#                                kwargs_lens=self._kwargs_lens)
#                point_amp = pointSource.image_amplitude(\
#                                kwargs_ps=self._kwargs_sourcefluxs,\
#                                kwargs_lens=self._kwargs_lens)

    def _update_image(self, kwargs_image):
        self.imaging_config = Data(kwargs_image)

    def _update_psf(self, kwargs_psf):
        self.psf = PSF(kwargs_psf)

    def compute(self):
        solver = LensEquationSolver(self.lensmodel)
        x_source, y_source = self.source.Lightlist[0]['x_center'],\
                            self.source.Lightlist[0]['y_center']

        theta_ra, theta_dec = solver.image_position_from_source(\
                                x_source, y_source, self._kwargs_lens,\
                                initial_guess_cut=False, search_window=10,\
                                min_distance=1e-2, num_iter_max=1e10,\
                                precision_limit=1e-5, verbose=True)

        return (theta_ra, theta_dec)

    def plot_lens_model(self):
        x_source, y_source = self.source.Lightlist[0]['x_center'],\
                            self.source.Lightlist[0]['y_center']
        print(x_source, y_source)

        f, axex = plt.subplots(1, 1, figsize=(5, 5))
        lens_plot.lens_model_plot(axex, lensModel=self.lensmodel,\
                                  kwargs_lens=self._kwargs_lens,\
                                  sourcePos_x=x_source, sourcePos_y=y_source,\
                                  point_source=True, with_caustics=True)
        plt.show()

    def image(self, kwargs_image, kwargs_psf, kwargs_numerics):
        self._update_image(kwargs_image)
        self._update_psf(kwargs_psf)

        psf = PSF(kwargs_psf)
        imageModel = ImageModel(data_class=self.imaging_config,\
                                psf_class=self.psf,\
                                lens_model_class=self.lensmodel,\
                                lens_light_model_class=self.lightmodel_lens,\
                                point_source_class=self.pointSource,\
                                kwargs_numerics=kwargs_numerics)

        image = imageModel.image(kwargs_lens=self._kwargs_lens,\
                                 kwargs_lens_light=self._kwargs_lensflux,\
                                 kwargs_ps=self._kwargs_psflux)
        return image

def main():
    lensG = oneobj(0.5, [{'Profile':'POINT_MASS', 'Mass':10,
                          'x_center':0., 'y_center':0.}],\
                        [{'Profile':'SERSIC', 'Flux':1, 'Re':0.5, 'n':1,\
                          'x_center':0., 'y_center':0.}])

    np.random.seed([1])
    x = np.random.uniform([-1,1])[0]
    y = np.random.uniform([-1,1])[0]

    lensQ = oneobj(5, [], [{'Profile':'PointSource', 'Flux': 1000,\
                            'x_center': x, 'y_center':y}])

    lensqso = LensQSO(lensG, lensQ)

    print(lensqso.compute())
    lensqso.plot_lens_model()
    deltaPix = 0.05

    kwargs_image = {'numPix': 200,
                'ra_at_xy_0': -5,
                'dec_at_xy_0': -5,
                'transform_pix2angle': np.array([[1, 0], [0, 1]]) * deltaPix}

    kwars_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.1, 'pixel_size': deltaPix}

    kwargs_numerics = {'subgrid_res': 50,
                    'psf_subgrid': True}

    image = lensqso.image(kwargs_image, kwars_psf, kwargs_numerics)

    plt.imshow(image)
    plt.show()




if __name__ == '__main__':
    main()
