###############################################################################
# (c) Copyright 2000-2021 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
#[=======================================================================[.rst:
Pr/PrConverters
---------------
#]=======================================================================]

gaudi_add_module(PrConverters
    SOURCES
        src/CompositesToV1Vertex.cpp
        src/SOATrackConverters.cpp
        src/TrackCompactVertexToV1Vertex.cpp
        src/fromLHCbTrackVector.cpp
        src/fromV1TrackV2Track.cpp
        src/fromV2TrackV1Track.cpp
        src/fromV3TrackV1Track.cpp
        src/fromVectorLHCbRecVertex.cpp
        src/fromVectorLHCbRecVertexv2.cpp
        src/fromVectorLHCbTrack.cpp
        src/fromPrTracksV1Track.cpp
        src/fromRecVertexv2RecVertexv1.cpp
        src/fromRecVertexv1RecVertexv2.cpp
        src/VPMicroClustersToVPLightClustersConverter.cpp
        src/VPLightClustersToVPHitsConverter.cpp
        src/VPLightClustersToVPMicroClustersConverter.cpp
        src/UTHitClustersToUTHitHandlerConverter.cpp
        src/UTHitClustersToPrUTHitsConverter.cpp
        src/UTHitHandlerToUTHitClusterConverter.cpp
        src/PrUTHitsToUTHitClusterConverter.cpp
        src/tests/TestOverlappingFTLiteClusters.cpp
        src/tests/TestOverlappingVPLightClusters.cpp
        src/tests/TestOverlappingUTHits.cpp
	    src/fromV2MuonPIDV1MuonPID.cpp
    LINK
        Gaudi::GaudiAlgLib
        Gaudi::GaudiKernel
        LHCb::EventBase
        LHCb::LHCbAlgsLib
        LHCb::LHCbKernel
        LHCb::LHCbMathLib
        LHCb::PhysEvent
        LHCb::RecEvent
        LHCb::TrackEvent
        LHCb::LinkerEvent
        Rec::PrKernel
        Rec::SelKernelLib
        Rec::TrackKernel
        Rec::TrackInterfacesLib
        LHCb::UTDAQLib
        LHCb::UTDetLib
        LHCb::VPDetLib
)

gaudi_add_tests(QMTest)
