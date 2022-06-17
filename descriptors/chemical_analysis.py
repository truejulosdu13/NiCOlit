import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from descriptors.dft_featurisation import *
from analysis import *


data_path = "data/utils/"

# featurization of the data-set directly with the dataframe for convenience of use in the visualization steps.
def feat_dft_nicolit(nicolit, data_path):
    # attention no precursor and no origin
    add_solvent_prop(nicolit, data_path)
    add_substrate_prop(nicolit, data_path)
    add_cp_prop(nicolit, data_path)
    add_lig_prop(nicolit, data_path)
    add_LA_prop(nicolit,data_path)
    nicolit.time = times(nicolit)
    nicolit.temperature = temperatures(nicolit)
    nicolit[['eq_substrate','eq_coupling_partner', 'eq_catalyst', 'eq_ligand','eq_reagent']] = equivalents(nicolit)
    choose_yield(nicolit)
    

# utils functions for feat_dft_nicolit
def add_solvent_prop(nicolit, data_path):
    solv = pd.read_csv(data_path + "solvents.csv", sep = ',', index_col=0)
    for prop in solv.columns:
        list_prop = [solv[prop][solvent] for solvent in nicolit.solvent]
        nicolit[prop] = list_prop
        
def add_substrate_prop(nicolit, data_path):
    substrate = pd.read_csv(data_path + "substrate_dft.csv", sep = ',', index_col=0)
    substrate.drop(columns=descritpors_to_remove_lig, inplace=True)
    canon_rdkit = [Chem.CanonSmiles(smi_co) for smi_co in substrate.index.to_list() ]
    substrate["can_rdkit"] = canon_rdkit
    substrate.set_index("can_rdkit", inplace=True)
    substrate = substrate[substrate.duplicated(keep='first') != True]
    substrate = substrate[~substrate.index.duplicated(keep='first')]
    for prop in substrate.columns:
        sub_prop =str("sub_"+prop)
        list_prop = [substrate[prop][solvent] for solvent in nicolit.substrate]
        nicolit[sub_prop] = list_prop

def add_cp_prop(nicolit, data_path):
    AX = pd.read_csv(data_path + "AX_dft.csv", sep = ',', index_col=0)
    AX.drop(columns=descritpors_to_remove_ax, inplace=True)
    canon_rdkit = [Chem.CanonSmiles(smi_co) for smi_co in AX.index.to_list() ]
    AX["can_rdkit"] = canon_rdkit
    AX.set_index("can_rdkit", inplace=True)
    for prop in AX.columns:
        ax_prop =str("ax_"+prop)
        list_prop = [AX[prop][solvent] for solvent in nicolit.effective_coupling_partner]
        nicolit[ax_prop] = list_prop

def add_lig_prop(nicolit, data_path):
        # issue : what should we put for nan ? 
    ligs = pd.read_csv(data_path + "ligand_dft.csv", sep = ',', index_col=0)
    ligs.drop(columns=descritpors_to_remove_lig, inplace=True)
    ligs.index.to_list()
    canon_rdkit = []
    for smi in ligs.index.to_list():
        try:
            canon_rdkit.append(Chem.CanonSmiles(smi))
        except:
            canon_rdkit.append(smi)
    ligs["can_rdkit"] = canon_rdkit
    ligs.set_index("can_rdkit", inplace=True)
    for prop in ligs.columns:
        lig_prop =str("lig_"+prop)
        list_prop = [ligs[prop][solvent] for solvent in nicolit.effective_ligand]
        nicolit[lig_prop] = list_prop
        
def add_LA_prop(nicolit, data_path):
    AL = pd.read_csv(data_path + "AL_dft.csv", sep = ',', index_col=0)
    AL.drop(columns=descritpors_to_remove_al, inplace=True)
    canon_rdkit = []
    for smi in AL.index.to_list():
        try:
            canon_rdkit.append(Chem.CanonSmiles(smi))
        except:
            canon_rdkit.append(smi)
    AL["can_rdkit"] = canon_rdkit
    AL.set_index("can_rdkit", inplace=True)
    for prop in AL.columns:
        al_prop =str("al_"+prop)
        list_prop = [AL[prop][solvent] for solvent in nicolit["Lewis Acid"]]
        nicolit[al_prop] = list_prop

def choose_yield(nicolit):
    yield_ = []
    for i, y in enumerate(nicolit.analytical_yield):
        if float(y) == float(y):
            float(y)
            yield_.append(float(y))
        else:
            yield_.append(float(nicolit.isolated_yield[i]))
    
    nicolit['yield'] = yield_



# Analysis of the performances on reduced parameters selection:
def perf(test_1, iterations=10):
    scores = []
    for i in range(iterations):
        X = test_1.drop(columns=['yield']).values
        y = test_1['yield']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        regr = RandomForestRegressor()
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    return np.mean(scores), np.std(scores)

def principal_descriptors(pi, L):
    restr_L   = []
    restr_imp = []
    restr_std = []
    for i,imp in enumerate(pi['importances_mean']):
        if imp > 0.01:
            restr_L.append(L[i])
            restr_imp.append(imp)
            restr_std.append(pi['importances_std'][i])
    plt.bar(restr_L, restr_imp, yerr = restr_std)
    plt.xticks(rotation=90)
    plt.show()
    return restr_L, restr_imp, restr_std

def perf_param(df, param_list):
    test = select_parameters(df, param_list)
    return perf(test, iterations=10)

def select_parameters(df, param_list):
    param_list.append('yield')
    return df[param_list]



# Tools for radar plots visualisation
# Stuff from matplotlib for spider plots

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

    
    
# lists of unnecessary DFT-descriptors:
descritpors_to_remove_lig = ["number_of_atoms", "charge", "multiplicity", "molar_mass", "molar_volume", "E_scf", "zero_point_correction", "E_thermal_correction","H_thermal_correction", "G_thermal_correction", "E_zpe", "E", "H", "G", "stoichiometry", "converged", "ES_root_molar_volume", "ES_root_electronic_spatial_extent",
    "X_0", "X_1", "X_2", "X_3", "X_4", "X_5", "X_6", "X_7",
    "Y_0", "Y_1", "Y_2", "Y_3", "Y_4", "Y_5", "Y_6", "Y_7", 
    "Z_0", "Z_1", "Z_2", "Z_3", "Z_4", "Z_5", "Z_6", "Z_7",
    "at_0", "at_1", "at_2", "at_3", "at_4", "at_5", "at_6", "at_7",                   'ES_root_Mulliken_charge_0','ES_root_Mulliken_charge_1','ES_root_Mulliken_charge_2','ES_root_Mulliken_charge_3','ES_root_Mulliken_charge_4','ES_root_Mulliken_charge_5','ES_root_Mulliken_charge_6',
'ES_root_Mulliken_charge_7',
'ES_root_NPA_charge_0','ES_root_NPA_charge_1', 'ES_root_NPA_charge_2', 'ES_root_NPA_charge_3', 'ES_root_NPA_charge_4', 'ES_root_NPA_charge_5','ES_root_NPA_charge_6','ES_root_NPA_charge_7',
 'ES_root_NPA_core_0', 'ES_root_NPA_core_1', 'ES_root_NPA_core_2', 'ES_root_NPA_core_3', 'ES_root_NPA_core_4', 'ES_root_NPA_core_5', 'ES_root_NPA_core_6', 'ES_root_NPA_core_7',
 'ES_root_NPA_valence_0', 'ES_root_NPA_valence_1', 'ES_root_NPA_valence_2', 'ES_root_NPA_valence_3', 'ES_root_NPA_valence_4', 'ES_root_NPA_valence_5', 'ES_root_NPA_valence_6', 'ES_root_NPA_valence_7',
 'ES_root_NPA_Rydberg_0', 'ES_root_NPA_Rydberg_1', 'ES_root_NPA_Rydberg_2', 'ES_root_NPA_Rydberg_3', 'ES_root_NPA_Rydberg_4', 'ES_root_NPA_Rydberg_5', 'ES_root_NPA_Rydberg_6', 'ES_root_NPA_Rydberg_7',
 'ES_root_NPA_total_0', 'ES_root_NPA_total_1', 'ES_root_NPA_total_2', 'ES_root_NPA_total_3', 'ES_root_NPA_total_4', 'ES_root_NPA_total_5', 'ES_root_NPA_total_6', 'ES_root_NPA_total_7',
 'ES_transition_0', 'ES_transition_1', 'ES_transition_2', 'ES_transition_3', 'ES_transition_4', 'ES_transition_5', 'ES_transition_6', 'ES_transition_7', 'ES_transition_8', 'ES_transition_9',
 'ES_osc_strength_0', 'ES_osc_strength_1', 'ES_osc_strength_2', 'ES_osc_strength_3', 'ES_osc_strength_4', 'ES_osc_strength_5', 'ES_osc_strength_6', 'ES_osc_strength_7', 'ES_osc_strength_8', 'ES_osc_strength_9',
 'ES_<S**2>_0', 'ES_<S**2>_1', 'ES_<S**2>_2', 'ES_<S**2>_3', 'ES_<S**2>_4', 'ES_<S**2>_5', 'ES_<S**2>_6', 'ES_<S**2>_7', 'ES_<S**2>_8','ES_<S**2>_9']

descritpors_to_remove_ax = ["number_of_atoms", "charge", "multiplicity", "molar_mass", "molar_volume", "E_scf", "zero_point_correction", "E_thermal_correction","H_thermal_correction", "G_thermal_correction", "E_zpe", "E", "H", "G", "stoichiometry", "converged", "ES_root_molar_volume", "ES_root_electronic_spatial_extent",
                        "X_0", "X_1", "X_2", "X_3",
                        "Y_0", "Y_1", "Y_2", "Y_3",
                        "Z_0", "Z_1", "Z_2", "Z_3",
                        "at_0", "at_1", "at_2", "at_3",
                        'ES_root_Mulliken_charge_0', 'ES_root_Mulliken_charge_1', 'ES_root_Mulliken_charge_2', 'ES_root_Mulliken_charge_3',
                         'ES_root_NPA_charge_0', 'ES_root_NPA_charge_1', 'ES_root_NPA_charge_2', 'ES_root_NPA_charge_3',
                         'ES_root_NPA_core_0', 'ES_root_NPA_core_1', 'ES_root_NPA_core_2', 'ES_root_NPA_core_3', 
                         'ES_root_NPA_valence_0', 'ES_root_NPA_valence_1', 'ES_root_NPA_valence_2', 'ES_root_NPA_valence_3',
                         'ES_root_NPA_Rydberg_0', 'ES_root_NPA_Rydberg_1', 'ES_root_NPA_Rydberg_2', 'ES_root_NPA_Rydberg_3',
                         'ES_root_NPA_total_0', 'ES_root_NPA_total_1', 'ES_root_NPA_total_2', 'ES_root_NPA_total_3',
                         'ES_transition_0', 'ES_transition_1', 'ES_transition_2', 'ES_transition_3', 'ES_transition_4', 'ES_transition_5', 'ES_transition_6', 'ES_transition_7', 'ES_transition_8', 'ES_transition_9',
                         'ES_osc_strength_0', 'ES_osc_strength_1', 'ES_osc_strength_2', 'ES_osc_strength_3', 'ES_osc_strength_4', 'ES_osc_strength_5', 'ES_osc_strength_6', 'ES_osc_strength_7', 'ES_osc_strength_8', 'ES_osc_strength_9',
                         'ES_<S**2>_0', 'ES_<S**2>_1', 'ES_<S**2>_2', 'ES_<S**2>_3', 'ES_<S**2>_4', 'ES_<S**2>_5', 'ES_<S**2>_6', 'ES_<S**2>_7', 'ES_<S**2>_8', 'ES_<S**2>_9']

descritpors_to_remove_al = ["converged", "stoichiometry", "ES_root_molar_volume", "X_0", "Y_0", "Z_0", "at_0", "ES_transition_7", "ES_transition_8", "ES_transition_9", 'ES_osc_strength_7', 'ES_osc_strength_8', 'ES_osc_strength_9', 'ES_<S**2>_7', 'ES_<S**2>_8', 'ES_<S**2>_9']