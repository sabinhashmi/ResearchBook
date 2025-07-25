###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
"""
Tests conversions of AoS-like Tracks and RecVertices among themselsves.

In/Out table of Track converters
 Converter                      In                              Out
  TrkConvKeyContV1ToVecV2       Event::v1::Tracks               std::vector<Event::v2::Track>
  TrkConvVecV2ToVecV1           std::vector<Event::v2::Track>   std::vector<Event::v1::Track>
  TrkConvVecV2ToKeyContV1       std::vector<Event::v2::Track>   Event::v1::Tracks
  TrkConvVecV1ToKeyContV1       std::vector<LHCb::Track>        LHCb::Tracks

In/Out table of RecVertex converters
 Converter                      In                              Out
  VtxConvKeyContV1ToVecV2       LHCb::RecVertices               LHCb::Event::v2::RecVertices
  VtxConvVecV2ToVecV1           LHCb::Event::v2::RecVertices    std::vector<LHCb::RecVertex>
  VtxConvVecV1ToVecV2           std::vector<LHCb::RecVertex>    LHCb::Event::v2::RecVertices
  VtxConvVecV2ToKeyContV1       LHCb::Event::v2::RecVertices    LHCb::RecVertices
  VtxConvVecV1ToKeyContV1       std::vector<LHCb::RecVertex>    LHCb::RecVertices

RecVertex conversion dataflow in this test
InFile -> LHCb::RecVertices
  -> VtxConvKeyContV1ToVecV2 -> LHCb::Event::v2::RecVertices
    -> VtxConvVecV2ToVecV1 -> std::vector<LHCb::RecVertex>
      -> VtxConvVecV1ToVecV2 -> Event::v2::RecVertices
        -> VtxConvVecV2ToKeyContV1 -> LHCb::RecVertices
      -> VtxConvVecV1ToKeyContV1 -> LHCb::RecVertices
"""

from PRConfig.TestFileDB import test_file_db
from PyConf.Algorithms import (
    LHCb__Converters__RecVertex__v1__fromRecVertexv2RecVertexv1 as VtxConvVecV2ToVecV1,
)
from PyConf.Algorithms import (
    LHCb__Converters__RecVertex__v1__fromVectorLHCbRecVertex as VtxConvVecV1ToKeyContV1,
)
from PyConf.Algorithms import (
    LHCb__Converters__RecVertex__v2__fromRecVertexv1RecVertexv2 as VtxConvVecV1ToVecV2,
)
from PyConf.Algorithms import (
    LHCb__Converters__RecVertex__v2__fromVectorLHCbRecVertices as VtxConvVecV2ToKeyContV1,
)
from PyConf.Algorithms import (
    LHCb__Converters__RecVertices__LHCbRecVerticesToVectorV2RecVertex as VtxConvKeyContV1ToVecV2,
)
from PyConf.Algorithms import (
    LHCb__Converters__Track__v1__fromV2TrackV1Track as TrkConvVecV2ToKeyContV1,
)
from PyConf.Algorithms import (
    LHCb__Converters__Track__v1__fromV2TrackV1TrackVector as TrkConvVecV2ToVecV1,
)
from PyConf.Algorithms import (
    LHCb__Converters__Track__v1__fromVectorLHCbTrack as TrkConvVecV1ToKeyContV1,
)
from PyConf.Algorithms import (
    LHCb__Converters__Track__v2__fromV1TrackV2Track as TrkConvKeyContV1ToVecV2,
)
from PyConf.Algorithms import PatPV3DFuture, UniqueIDGeneratorAlg, UnpackTrack
from PyConf.application import configure, configure_input, input_from_root_file
from PyConf.components import force_location
from PyConf.control_flow import CompositeNode, NodeLogic


def fromFileToKeyContV1(options):
    """
    Create KeyedContainer's of Event::v1::Track and Event::v1::RecVertex
    from the input file.
    The Tracks are the unpacked Rec/Track/Best tracks.
    The RecVertices are created by PatPV3DFuture from the Tracks.
    """
    track_alg = UnpackTrack(
        InputName=input_from_root_file(
            "pRec/Track/Best", forced_type="LHCb::PackedTracks", options=options
        )
    )
    tracks = track_alg.OutputName
    vtx_alg = PatPV3DFuture(InputTracks=tracks)

    return {
        "algs": [track_alg, vtx_alg],
        "data": {"tracks": tracks, "vertices": vtx_alg.OutputVerticesName},
    }


class fromToConverter(object):
    """
    Callable class for creating and configuring a matchd pair of Track and
    RecVertex converters.
    """

    def __init__(self, t_conv_type, v_conv_type, inputNames=True):
        """
        Constructor
        Inuts:
          t_conv_type:  Configurable
              Configurable class for the intended Track converter
          v_conv_type:  Configurable
              Configurable class for the intended RecVertex converter
          inputNames:  bool
                Flag for handling a special case in which the parameter
                names for the vertex converted does not end in 'Name'.
        """
        self._tct = t_conv_type
        self._vct = v_conv_type
        self._inNames = inputNames

        return

    def __call__(self, tracks, vertices):
        """
        Maker for the pair of algorithms.
        Inputs:
          tracks:  DataHandle
                DataHandle for the input tracks
          vertices:  DataHandle
                DataHandle for the input vertices
        Returns:
          Dictionary containing two items:
            'args':  Ordered list of configured algorithms
            'data':  Dictionary of DataHandles for the converted 'tracks' and
                     'vertices'.
        """
        track_alg = self._tct(InputTracksName=tracks)
        out_tracks = track_alg.OutputTracksName
        if self._inNames:
            vtx_alg = self._vct(InputTracksName=out_tracks, InputVerticesName=vertices)
            out_vtxs = vtx_alg.OutputVerticesName
        else:
            vtx_alg = self._vct(InputTracks=out_tracks, InputVertices=vertices)
            out_vtxs = vtx_alg.OutputVertices

        return {
            "algs": [track_alg, vtx_alg],
            "data": {"tracks": out_tracks, "vertices": out_vtxs},
        }


options = test_file_db["upgrade_minbias_hlt1_filtered"].make_lbexec_options(
    evt_max=100,
    output_file="output.root",
)
config = configure_input(options)

# Construct the conversion steps
v1Keyed = fromFileToKeyContV1(options)

fromKeyContV1ToVectorV2 = fromToConverter(
    TrkConvKeyContV1ToVecV2, VtxConvKeyContV1ToVecV2, inputNames=False
)
v2Vect = fromKeyContV1ToVectorV2(**v1Keyed["data"])

fromVectorV2ToVectorV1 = fromToConverter(TrkConvVecV2ToVecV1, VtxConvVecV2ToVecV1)
v1Vect = fromVectorV2ToVectorV1(**v2Vect["data"])

# No direct track converter, needs to be done in two steps
fromVectorV1ToVectorV2 = fromToConverter(TrkConvKeyContV1ToVecV2, VtxConvVecV1ToVecV2)
v2Vect_2 = fromVectorV1ToVectorV2(
    TrkConvVecV1ToKeyContV1(InputTracksName=v1Vect["data"]["tracks"]).OutputTracksName,
    vertices=v1Vect["data"]["vertices"],
)

fromVectorV2ToKeyContV1 = fromToConverter(
    TrkConvVecV2ToKeyContV1, VtxConvVecV2ToKeyContV1
)
v1Keyed_2 = fromVectorV2ToKeyContV1(**v2Vect_2["data"])

fromVectorV1ToKeyContV1 = fromToConverter(
    TrkConvVecV1ToKeyContV1, VtxConvVecV1ToKeyContV1
)
v1Keyed_3 = fromVectorV1ToKeyContV1(**v1Vect["data"])

prec_node = CompositeNode(
    "precursors", combine_logic=NodeLogic.LAZY_AND, children=[UniqueIDGeneratorAlg()]
)

conv_node = CompositeNode(
    "finalconverters",
    combine_logic=NodeLogic.NONLAZY_AND,
    children=v1Keyed_2["algs"] + v1Keyed_3["algs"],
)

top_node = CompositeNode(
    "top",
    combine_logic=NodeLogic.LAZY_AND,
    children=[prec_node, conv_node],
    force_order=True,
)

config.update(configure(options, top_node))
