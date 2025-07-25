# Moore Configuration
from Moore import options, run_moore
from Moore.config import register_line_builder
from Moore.lines import Hlt2Line

import Functors as F
from Functors.math import in_range
from GaudiKernel.SystemOfUnits import MeV, mm 

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


dump_track_type = 'Downstream'
eval_track_type1 = 'Seed'
eval_track_type2 = 'Downstream'

# Function to dump tracker data
def tracker_dumper(odin_location=make_odin, velo_hits=make_VPClus_hits, with_ut=True):
    ut_hits = make_PrStoreUTHit_hits() if with_ut else PrStoreUTHitEmptyProducer().Output
    track_input_location = make_hlt2_tracks()[dump_track_type]
    if isinstance(track_input_location, dict) and "v1" in track_input_location:
        track_input_location = track_input_location["v1"]

    links_to_lhcbids = (
        make_links_tracks_mcparticles(
            InputTracks=track_input_location, 
            LinksToLHCbIDs=make_links_lhcbids_mcparticles_tracking_system()
        ) if with_ut else make_links_lhcbids_mcparticles_VP_FT()
    )
    
    return PrTrackRecoDumper(
        TrackLocation=track_input_location, 
        MCParticlesLocation=mc_unpackers()["MCParticles"],
        VPLightClusterLocation=velo_hits(),
        ODINLocation=odin_location(),
        FTHitsLocation=make_PrStoreSciFiHits_hits(),
        UTHitsLocation=ut_hits,
        LinksLocation=links_to_lhcbids,
    )


def efficiency_checker(track_type):
    input_tracks_config = make_hlt2_tracks()[track_type]
    if isinstance(input_tracks_config, dict) and 'v1' in input_tracks_config:
        input_tracks = input_tracks_config['v1']
    else:
        input_tracks = input_tracks_config

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
def downstream_line(name="Hlt2DebugTracking", prescale=1, persistreco=True):
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


public_tools = [stateProvider_with_simplified_geom()]


def make_lines():
    return [line_builder() for line_builder in all_lines.values()]

# Moore execution options
options.input_files = [
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000005_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000008_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000013_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000018_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000022_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000023_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000030_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000037_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000045_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000050_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000054_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000058_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000065_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000066_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000069_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000072_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000075_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000081_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000082_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000086_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000087_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000089_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000095_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000097_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000099_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000104_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000105_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000106_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000108_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000112_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000115_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000119_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000120_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000122_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000123_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000125_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000132_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000133_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000136_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000140_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000144_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000147_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000148_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000153_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000158_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000160_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000163_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000167_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000169_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000173_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000175_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000178_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000183_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000184_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000186_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000189_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000192_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000193_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000198_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000200_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000203_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000206_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000211_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000213_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000215_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000221_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000222_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000226_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000227_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000233_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000236_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000239_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000241_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000243_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000247_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000248_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000253_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000254_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000256_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000259_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000263_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000264_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000267_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000272_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000273_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000276_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000279_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000283_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000285_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000291_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000292_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000295_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000297_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000298_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000303_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000307_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000308_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000311_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000313_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000315_1.digi',
    'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/2024/DIGI/00230966/0000/00230966_00000321_1.digi',
]


options.input_type = "ROOT"
options.input_raw_format = 0.5
options.persistreco_version = 0.0
options.evt_max = 1000
options.simulation = True
options.data_type = "Upgrade"
options.dddb_tag = 'dddb-20240427'
options.conddb_tag = 'sim10-2024.Q1.2-v1.1-md100'
options.root_ioalg_name = "RootIOAlgExt"
options.root_ioalg_opts = {"IgnorePaths": ["/Event/Rec/Summary"]}
# options.output_file = 'MooreOutput.dst'
options.output_type = 'ROOT'
# options.output_manifest_file = "MooreManifest.tck.json"
# options.control_flow_file = "ControlFlow.gv"
# options.data_flow_file = "DataFlow.gv"
# options.histo_file = "MooreHistogram.root"
# options.ntuple_file = "MooreTuple.root"
options.n_threads=-1
with reconstruction.bind(from_file=False):
    run_moore(options, make_lines, public_tools)