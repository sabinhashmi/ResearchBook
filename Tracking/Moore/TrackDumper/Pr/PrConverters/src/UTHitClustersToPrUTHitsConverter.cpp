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

#include <Event/PrHits.h>
#include <Event/UTHitCluster.h>

#include <UTDet/DeUTDetector.h>
#include <UTDet/DeUTSector.h>

#include "LHCbAlgs/Transformer.h"

namespace LHCb {
  /**
   *  Transform the "UTHitClusters" to a container of PrUTHits (SoA).
   *
   *  Depends on the geometry by the filling of the cluster's position information
   *
   *  @author Laurent Dufour
   */
  struct UTHitClustersToPrUTHitsConverter
      : Algorithm::Transformer<::LHCb::Pr::UT::Hits( const EventContext&, UTHitClusters const&, DeUTDetector const& ),
                               Algorithm::Traits::usesConditions<DeUTDetector>> {

    UTHitClustersToPrUTHitsConverter( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator,
                       { KeyValue{ "InputClusters", "" }, KeyValue{ "DeUTLocation", DeUTDetLocation::location() } },
                       { KeyValue{ "OutputClusters", "" } } ) {}
    ::LHCb::Pr::UT::Hits operator()( const EventContext&, UTHitClusters const&, DeUTDetector const& ) const override;

    mutable Gaudi::Accumulators::AveragingCounter<unsigned long> m_convertedClusters{ this, "# Converted Clusters" };
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::UTHitClustersToPrUTHitsConverter, "UTHitClustersToPrUTHitsConverter" )

::LHCb::Pr::UT::Hits LHCb::UTHitClustersToPrUTHitsConverter::operator()( const EventContext&        evtCtx,
                                                                         LHCb::UTHitClusters const& dst,
                                                                         DeUTDetector const&        deut ) const {

  ::LHCb::Pr::UT::Hits hitHandler{ Zipping::generateZipIdentifier(), LHCb::getMemResource( evtCtx ) };
  hitHandler.reserve( dst.size() );

  for ( const auto& ut_hit_cluster : dst ) {
    const auto& sector = deut.getSector( ut_hit_cluster.channelID() );
    // tested with assert( sector == deut.findSector( ut_hit_cluster.channelID() ) );
    // (on dd4hep, on detdesc you'd need to dereference findSector)

    const auto& refSector = sector;

    hitHandler.emplace_back( refSector, ut_hit_cluster.channelID().sectorFullID(), ut_hit_cluster.strip(),
                             ut_hit_cluster.fracStrip(), ut_hit_cluster.channelID(), ut_hit_cluster.size(), false,
                             ut_hit_cluster.clusterCharge() );
  }

  m_convertedClusters += hitHandler.nHits();
  hitHandler.addPadding();

  return hitHandler;
}
