/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "Event/UTHitCluster.h"
#include <PrKernel/UTHitHandler.h>

#include <UTDAQ/UTDAQHelper.h>
#include <UTDet/DeUTDetector.h>
#include <UTDet/DeUTSector.h>

#include "LHCbAlgs/Transformer.h"

namespace LHCb {
  /**
   *  Transform the "UTHitClusters" to a UT::Hits (AoS) container.
   *
   *  The standard use-case for this is the interpretation of persisted
   *  (bare) UT clusters information on the tracks, as a normal
   *  cluster container as if created by the decoding.
   *
   *  Depends on the geometry by the filling of the cluster's position information
   *
   *  @author Laurent Dufour
   */
  struct UTHitClustersToUTHitHandlerConverter
      : Algorithm::Transformer<::UT::HitHandler( const EventContext&, UTHitClusters const&, DeUTDetector const& ),
                               Algorithm::Traits::usesConditions<DeUTDetector>> {

    UTHitClustersToUTHitHandlerConverter( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator,
                       { KeyValue{ "InputClusters", "" }, KeyValue{ "DeUTLocation", DeUTDetLocation::location() } },
                       { KeyValue{ "OutputClusters", "" } } ) {}
    ::UT::HitHandler operator()( const EventContext&, UTHitClusters const&, DeUTDetector const& ) const override;

    mutable Gaudi::Accumulators::AveragingCounter<unsigned long> m_convertedClusters{ this, "# Converted Clusters" };
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::UTHitClustersToUTHitHandlerConverter, "UTHitClustersToUTHitHandlerConverter" )

::UT::HitHandler LHCb::UTHitClustersToUTHitHandlerConverter::operator()( const EventContext&        evtCtx,
                                                                         LHCb::UTHitClusters const& dst,
                                                                         DeUTDetector const&        deut ) const {

  ::UT::HitHandler hitHandler{ Zipping::generateZipIdentifier(), LHCb::getMemResource( evtCtx ) };
  hitHandler.reserve( dst.size() );

  for ( const auto& ut_lite_cluster : dst ) {
    const auto& sector = deut.findSector( ut_lite_cluster.channelID() );

#ifdef USE_DD4HEP
    const auto& refSector = sector;
#else
    const auto& refSector = *sector;
#endif

    hitHandler.emplace_back( refSector, ut_lite_cluster.channelID().sectorFullID(), ut_lite_cluster.strip(),
                             ut_lite_cluster.fracStrip(), ut_lite_cluster.channelID(), ut_lite_cluster.size(), false,
                             ut_lite_cluster.clusterCharge() );
  }

  m_convertedClusters += hitHandler.nbHits();

  return hitHandler;
}
