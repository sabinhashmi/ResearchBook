###############################################################################
# (c) Copyright 2000-2022 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
#[=======================================================================[.rst:
Pr/PrAlgorithms
---------------
#]=======================================================================]
gaudi_add_module(PrAlgorithms
    SOURCES
        src/PrAddUTHitsTool.cpp
        src/PrIgnoreUTHitsTool.cpp
        src/PrForwardTracking.cpp
        src/PrHybridSeeding.cpp
        src/PrLongLivedTracking.cpp
        src/PrMakeSelection.cpp
        src/PrMatchNN.cpp
        src/PrResidualPrUTHits.cpp
        src/PrResidualSciFiHits.cpp
        src/PrResidualSeeding.cpp
        src/PrResidualVeloTracks.cpp
        src/PrStoreSciFiHits.cpp
        src/PrStoreUTHit.cpp
        src/HitsEmptyProducer.cpp
    LINK
        Boost::container
        Boost::headers
        Gaudi::GaudiAlgLib
        Gaudi::GaudiKernel
        LHCb::LHCbAlgsLib
        LHCb::DAQEventLib
        LHCb::DetDescLib
        LHCb::FTDAQLib
        LHCb::FTDetLib
        LHCb::FTEvent
        LHCb::LHCbKernel
        LHCb::LHCbMathLib
        LHCb::LoKiCoreLib
        LHCb::MagnetLib
        LHCb::PhysEvent
        LHCb::RecEvent
        LHCb::TrackEvent
        LHCb::UTDAQLib
        LHCb::UTDetLib
        LHCb::UTKernelLib
        LHCb::VPDetLib
        Rec::PrKernel
        Rec::TrackInterfacesLib
        ROOT::MathCore
        Vc::Vc
        VDT::vdt
)
