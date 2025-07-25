###############################################################################
# (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
"""
Minimal module for processing KS0 particles from HLT2 output
"""

from PyConf.reading import get_particles
from FunTuple import FunTuple_Particles as Funtuple
from FunTuple import FunctorCollection
import Functors as F
from DaVinci import Options, make_config
from Configurables import MessageSvc
from GaudiKernel.Constants import INFO

def main(options: Options):
    print(f"Processing file: {options.input_files[0]}")
    
    # Set verbose level
    MessageSvc().OutputLevel = INFO
    
    # We know this location works based on previous tests
    ks_location = "/Event/HLT2/Hlt2PersistReco_Particles/Particles"
    print(f"Using particle location: {ks_location}")
    
    # Get particles
    particles = get_particles(ks_location)
    print(f"Found particles at: {ks_location}")
    
    # Define the decay descriptor
    fields = {
        "Ks": "KS0 -> pi+ pi-",
        "piplus": "KS0 -> ^pi+ pi-",
        "piminus": "KS0 -> pi+ ^pi-",
    }
    
    # Absolute minimal functors for KS0 - focused on mass
    ks_variables = FunctorCollection({
        "MASS": F.MASS,
        "PT": F.PT,
        "P": F.P
    })
    
    # Minimal functors for pions
    daughter_variables = FunctorCollection({
        "PT": F.PT,
        "P": F.P
    })
    
    # Combine the functors
    variables = {
        "Ks": ks_variables,
        "piplus": daughter_variables,
        "piminus": daughter_variables,
    }
    
    # Create the tuple
    my_tuple = Funtuple(
        name="KS0Tuple",
        tuple_name="DecayTree",
        fields=fields,
        variables=variables,
        inputs=particles
    )
    
    # Return the config
    return make_config(options, [my_tuple])