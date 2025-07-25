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

#include "Detector/VP/VPChannelID.h"
#include "Event/PrHits.h"
#include "Event/PrVeloHits.h"
#include "Event/VPLightCluster.h"
#include "LHCbAlgs/Transformer.h"
namespace {
  using VPHits = LHCb::Pr::VP::Hits;
}
namespace LHCb {

  /**
   *  Transform the "VPLightClusters" to VP::Hits.
   *
   *  @author Laurent Dufour
   */
  struct VPLightClustersToVPHitsConverter : Algorithm::Transformer<VPHits( VPLightClusters const& )> {

    VPLightClustersToVPHitsConverter( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator, { KeyValue{ "InputClusters", LHCb::VPClusterLocation::Light } },
                       { KeyValue{ "OutputHits", "" } } ) {}
    VPHits operator()( VPLightClusters const& ) const override;

    mutable Gaudi::Accumulators::AveragingCounter<unsigned long> m_convertedClusters{ this, "# Converted Clusters" };
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::VPLightClustersToVPHitsConverter, "VPLightClustersToVPHitsConverter" )

VPHits LHCb::VPLightClustersToVPHitsConverter::operator()( LHCb::VPLightClusters const& clusters ) const {
  VPHits hits;

  hits.reserve( clusters.size() );
  for ( const auto& vp_light_cluster : clusters ) {
    auto hit = hits.emplace_back<SIMDWrapper::InstructionSet::Scalar>();
    hit.field<LHCb::Pr::VP::VPHitsTag::pos>().set(
        { vp_light_cluster.x(), vp_light_cluster.y(), vp_light_cluster.z() } );
    hit.field<LHCb::Pr::VP::VPHitsTag::ChannelId>().set( SIMDWrapper::scalar::int_v( vp_light_cluster.channelID() ) );
  }

  m_convertedClusters += clusters.size();

  return hits;
}
