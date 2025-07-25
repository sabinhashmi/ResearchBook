# Moore Configuration
from Moore import options, run_moore
from Moore.config import register_line_builder
from Moore.lines import Hlt2Line

import Functors as F
from Functors.math import in_range  # Correct import for in_range
from Functors import require_all
from GaudiKernel.SystemOfUnits import GeV, MeV

from RecoConf.algorithms_thor import ParticleCombiner, ParticleFilter
from RecoConf.reconstruction_objects import make_pvs, upfront_reconstruction, reconstruction
from RecoConf.standard_particles import make_has_rich_long_pions
from RecoConf.event_filters import require_pvs
from RecoConf.global_tools import stateProvider_with_simplified_geom
from RecoConf.hlt2_tracking import (
    make_PrStoreUTHit_hits, make_VPClus_hits, make_PrStoreSciFiHits_hits, make_hlt2_tracks
)
from RecoConf.mc_checking import (
    make_links_lhcbids_mcparticles_tracking_system, make_links_lhcbids_mcparticles_VP_FT,
    make_links_tracks_mcparticles, check_tracking_efficiency
)
from RecoConf.mc_checking_categories import get_mc_categories, get_hit_type_mask
from RecoConf.muonid import make_muon_hits
from RecoConf.data_from_file import mc_unpackers
from RecoConf.decoders import default_ft_decoding_version

from PyConf.application import make_odin
from PyConf.Algorithms import PrStoreUTHitEmptyProducer, PrTrackRecoDumper
from RecoConf.calorimeter_reconstruction import make_digits

# Global Configurations
default_ft_decoding_version.global_bind(value=6)
make_muon_hits.global_bind(geometry_version=3)
make_digits.global_bind(calo_raw_bank=False)

all_lines = {}

# Function to filter pions based on transverse momentum (pT) and impact parameter
def filter_pions(particles, pvs, pt_min=300 * MeV, minipchi2_min=2):
    """
    Filter downstream pion candidates with minimum transverse momentum (pT) and IP chi2.
    
    :param particles: Input container of downstream track candidates.
    :param pvs: Primary vertices container (used for MINIPCHI2 computation).
    :param pt_min: Minimum transverse momentum threshold (default: 300 MeV).
    :param minipchi2_min: Minimum IP chi2 with respect to PV (default: 2).
    :return: Filtered pion container.
    """
    cut = F.require_all(
        F.PT > pt_min,  # pT cut to reduce soft background
        F.MINIPCHI2(pvs) > minipchi2_min
    )
    return ParticleFilter(
        Input=particles,
        Cut=F.FILTER(cut)
    )

# Function to create KS0 particles from filtered pions
def make_ks0(pions, pvs, mass_min=420 * MeV, mass_max=560 * MeV, sum_pt_min=800 * MeV,  
             vchi2pdof_max=50.0, bpvfdchi2_min=10.0, bpvdira_min=0.9):
    """
    Create KS0 candidates by combining pairs of downstream pions with kinematic,
    vertex quality, and displacement cuts.
    
    :param pions: Container of filtered downstream pion candidates.
    :param pvs: Primary vertices container.
    :param mass_min: Minimum invariant mass of the pi+pi- pair (default: 420 MeV).
    :param mass_max: Maximum invariant mass of the pi+pi- pair (default: 560 MeV).
    :param sum_pt_min: Minimum sum of pion pTs (default: 800 MeV).
    :param vchi2pdof_max: Maximum vertex chi2 per degree of freedom (default: 50).
    :param bpvfdchi2_min: Minimum flight distance chi2 from the PV (default: 10).
    :param bpvdira_min: Minimum pointing cosine angle (BPVDIRA) with respect to the PV (default: 0.9).
    :param fd_min: Minimum flight distance from the PV in mm (default: 5 mm).
    :return: A ParticleCombiner configured for KS0 reconstruction.
    """
    combination_cut = F.require_all(
        in_range(mass_min, F.MASS, mass_max),  # Mass window cut for KS0 candidates
        F.SUM(F.PT) > sum_pt_min,              # Sum of pion pTs to reduce soft background
    )
    composite_cut = F.require_all(
        F.CHI2DOF < vchi2pdof_max,             # Vertex chi2/dof cut
        F.BPVFDCHI2(pvs) > bpvfdchi2_min,        # Flight distance chi2 cut
        F.BPVDIRA(pvs) > bpvdira_min           # Directional cosine cut (pointing to PV)
    )
    return ParticleCombiner(
        Inputs=[pions, pions],
        DecayDescriptor="KS0 -> pi+ pi-",
        CombinationCut=combination_cut,
        CompositeCut=composite_cut,
    )

@register_line_builder(all_lines)
def kshort_line(name="Hlt2PersistReco_Particles", prescale=1, persistreco=True):
    """
    Defines the HLT2 reconstruction line for KS0 candidates using downstream tracks.
    
    This line performs the following:
      1. Creates primary vertices (PVs).
      2. Filters downstream pion candidates using kinematic and IP cuts.
      3. Combines the filtered pions to create KS0 candidates with defined invariant mass, 
         pT, vertex quality, and displacement requirements.
      4. Applies upstream reconstruction and required PV selections.
    
    :param name: Name of the HLT2 line.
    :param prescale: Prescale factor for the line.
    :param persistreco: Boolean flag for persistent reconstruction.
    :return: A configured Hlt2Line for persistent reconstruction.
    """
    # Create a single primary vertex container
    pvs = make_pvs()
    pions = filter_pions(make_has_rich_long_pions(), pvs)
    ks0 = make_ks0(pions, pvs)
    
    # Build the line sequence using upfront reconstruction and PV filtering
    alg_sequence = upfront_reconstruction() + [require_pvs(pvs)] + [ks0]
    
    return Hlt2Line(
        name=name,
        algs=alg_sequence,
        prescale=prescale,
        persistreco=persistreco,
    )

# Define track types
dump_track_type = 'Downstream'
eval_track_type1 = 'Seed'
eval_track_type2 = 'Downstream'

# Function to dump tracker data
def tracker_dumper(odin_location=make_odin, velo_hits=make_VPClus_hits, with_ut=True):
    ut_hits = make_PrStoreUTHit_hits() if with_ut else PrStoreUTHitEmptyProducer().Output
    links_to_lhcbids = (
        make_links_tracks_mcparticles(
            InputTracks=make_hlt2_tracks()[dump_track_type],
            LinksToLHCbIDs=make_links_lhcbids_mcparticles_tracking_system()
        ) if with_ut else make_links_lhcbids_mcparticles_VP_FT()
    )
    
    return PrTrackRecoDumper(
        TrackLocation=make_hlt2_tracks()[dump_track_type]["v1"],
        MCParticlesLocation=mc_unpackers()["MCParticles"],
        VPLightClusterLocation=velo_hits(),
        ODINLocation=odin_location(),
        FTHitsLocation=make_PrStoreSciFiHits_hits(),
        UTHitsLocation=ut_hits,
        LinksLocation=links_to_lhcbids,
    )

# Function to check tracking efficiency
def efficiency_checker(track_type):
    input_tracks = make_hlt2_tracks()[track_type]['v1']
    return check_tracking_efficiency(
        TrackType=track_type,
        InputTracks=input_tracks,
        LinksToTracks=make_links_tracks_mcparticles(
            InputTracks=input_tracks,
            LinksToLHCbIDs=make_links_lhcbids_mcparticles_tracking_system()
        ),
        LinksToLHCbIDs=make_links_lhcbids_mcparticles_tracking_system(),
        MCCategories=get_mc_categories(track_type),
        HitTypesToCheck=get_hit_type_mask(track_type)
    )
    
@register_line_builder(all_lines)
def downstream_line(name="Hlt2PersistReco_Track", prescale=1, persistreco=True):
    """
    Defines the HLT2 line for tracking reconstruction for downstream and seed tracks.
    
    This line:
      1. Creates primary vertices (PVs) once.
      2. Checks tracking efficiency for the specified track types.
      3. Dumps the tracker information.
    
    :param name: Name of the HLT2 line.
    :param prescale: Prescale factor for the line.
    :param persistreco: Boolean flag for persistent reconstruction.
    :return: A configured Hlt2Line for tracking reconstruction.
    """
    # Create a single PV container to be used by the entire line
    pvs = make_pvs()
    
    alg_sequence = (
        upfront_reconstruction() +
        [require_pvs(pvs)] +
        [efficiency_checker(eval_track_type1)] +
        [efficiency_checker(eval_track_type2)] +
        [tracker_dumper()]
    )
    
    return Hlt2Line(
        name=name,
        algs=alg_sequence,
        prescale=prescale,
        persistreco=persistreco,
    )

# Public tools configuration
public_tools = [stateProvider_with_simplified_geom()]

# Function to create all lines
def make_lines():
    return [line_builder() for line_builder in all_lines.values()]

# Moore execution options
options.input_files = ['root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000236_1.digi']
options.input_type = "ROOT"
options.input_raw_format = 0.5
options.persistreco_version = 0.0
options.evt_max = 1000
options.simulation = True
options.data_type = "Upgrade"
options.dddb_tag = 'dddb-20240427'
options.conddb_tag = 'sim10-2024.Q1.2-v1.1-md100'
options.geometry_version = 'trunk'
options.conditions_version = 'master'
options.root_ioalg_name = "RootIOAlgExt"
options.root_ioalg_opts = {"IgnorePaths": ["/Event/Rec/Summary"]}
options.output_file = 'MooreOutput.dst'
options.output_type = 'ROOT'
options.output_manifest_file = "MooreManifest.tck.json"
options.control_flow_file = "ControlFlow.gv"
options.data_flow_file = "DataFlow.gv"
options.histo_file = "MooreHistogram.root"
options.ntuple_file = "MooreTuple.root"

# Run Moore with the configured options
with reconstruction.bind(from_file=False):
    run_moore(options, make_lines, public_tools)