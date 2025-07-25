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
#include <Event/PrHits.h>

#include "LHCbAlgs/Transformer.h"

#include "Detector/UT/UTConstants.h"

namespace LHCb {
  /**
   *  Transform the SoA PrUTHits to a set of UTHitClusters.
   *
   *  Effectively removing any geometry-dependent information from
   *  the hits, leaving only the bare minimum to reproduce these
   *  hits given a geometry, and going to a finite numerical
   *  precision.
   *
   *  @author Laurent Dufour
   */
  struct PrUTHitsToUTHitClusterConverter : Algorithm::Transformer<UTHitClusters( ::LHCb::Pr::UT::Hits const& )> {

    PrUTHitsToUTHitClusterConverter( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator, { KeyValue{ "InputHits", "" } }, { KeyValue{ "OutputClusters", "" } } ) {}
    UTHitClusters operator()( ::LHCb::Pr::UT::Hits const& ) const override;

    mutable Gaudi::Accumulators::AveragingCounter<unsigned long> m_convertedClusters{ this, "# Converted Clusters" };
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::PrUTHitsToUTHitClusterConverter, "PrUTHitsToUTHitClusterConverter" )

LHCb::UTHitClusters LHCb::PrUTHitsToUTHitClusterConverter::operator()( ::LHCb::Pr::UT::Hits const& dst ) const {
  LHCb::UTHitClusters outputClusters;
  outputClusters.reserve( dst.nHits() );

  for ( unsigned int uniqueSector = 0; uniqueSector < LHCb::Detector::UT::MaxNumberOfSectors; ++uniqueSector ) {
    const auto& indices = dst.indices( uniqueSector );
    if ( indices.first == indices.second ) continue;

    for ( int i = indices.first; i != indices.second; i++ ) {
      const auto&                   hit = dst.scalar()[i];
      LHCb::Detector::UT::ChannelID channelID( hit.get<LHCb::Pr::UT::UTHitsTag::channelID>().cast() );
      const double                  fracStrip     = hit.get<LHCb::Pr::UT::UTHitsTag::fracStrip>().cast();
      const unsigned int            clusterSize   = hit.get<LHCb::Pr::UT::UTHitsTag::clusterSize>().cast();
      const auto                    clusterCharge = hit.get<LHCb::Pr::UT::UTHitsTag::clusterCharge>().cast();

      outputClusters.addHit( channelID, fracStrip, clusterSize, clusterCharge );
    }
  }

  outputClusters.setOffsets();

  m_convertedClusters += outputClusters.size();

  return outputClusters;
}
