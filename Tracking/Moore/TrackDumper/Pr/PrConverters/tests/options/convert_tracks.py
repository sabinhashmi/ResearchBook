###############################################################################
# (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
"""
Test the conversion of AoS-like tracks aka (v1::Track and v2::Track) among
themselves and to SoA track containers.
"""

from Gaudi.Configuration import ERROR
from PyConf.Algorithms import (
    LHCb__Converters__Track__SOA__fromV1Track as fromV1TrackFittedGenericTrack,
)
from PyConf.Algorithms import (
    LHCb__Converters__Track__v1__fromV2TrackV1Track as fromV2TrackV1Track,
)
from PyConf.Algorithms import (
    LHCb__Converters__Track__v2__fromV1TrackV2Track as fromV1TrackV2Track,
)
from PyConf.Algorithms import UniqueIDGeneratorAlg, UnpackTrack
from PyConf.application import (
    ApplicationOptions,
    configure,
    configure_input,
    input_from_root_file,
)
from PyConf.control_flow import CompositeNode

options = ApplicationOptions(_enabled=False)
# Pick a file that has the reconstruction available
options.set_input_and_conds_from_testfiledb("upgrade_minbias_hlt1_filtered")
options.evt_max = 100
config = configure_input(options)

# Unpack tracks
track_unpacker = UnpackTrack(
    InputName=input_from_root_file("pRec/Track/Best", forced_type="LHCb::PackedTracks")
)

# Track converters (old versions)
track_converter_v1_to_v2 = fromV1TrackV2Track(
    name="TrackConverter_v1_to_v2", InputTracksName=track_unpacker.OutputName
)
track_converter_v2_to_v1 = fromV2TrackV1Track(
    name="TrackConverter_v2_to_v1",
    InputTracksName=track_converter_v1_to_v2.OutputTracksName,
)

idgen = UniqueIDGeneratorAlg(name="UniqueIDGeneratorAlg")

# Track converters (SIMD-friendly version)
track_converter_v1_to_SOA = fromV1TrackFittedGenericTrack(
    name="TrackConverter_v1_to_SOA",
    InputTracks=track_converter_v2_to_v1.OutputTracksName,
    InputUniqueIDGenerator=idgen.Output,
    RestrictToType="Long",
    OutputLevel=ERROR,
)  # avoid invalid states warnings

# Define the sequence
algs = [
    track_converter_v2_to_v1,
    # see issue #410
    # track_converter_v1_to_SOA
]

config.update(configure(options, CompositeNode("test", algs)))
