/*****************************************************************************\
* (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "Detector/VP/VPChannelID.h"
#include "Event/VPLightCluster.h"
#include "Event/VPMicroCluster.h"
#include "LHCbAlgs/Transformer.h"

namespace LHCb {

  /**
   *  Transform the "VPLightClusters" to a VPMicroClusters,
   *  i.e. excluding the x/y/z information.
   *
   *  @author Laurent Dufour
   */
  struct VPLightClustersToVPMicroClustersConverter : Algorithm::Transformer<VPMicroClusters( VPLightClusters const& )> {

    VPLightClustersToVPMicroClustersConverter( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator, { KeyValue{ "InputClusters", LHCb::VPClusterLocation::Light } },
                       { KeyValue{ "OutputClusters", LHCb::VPClusterLocation::Micro } } ) {}
    LHCb::VPMicroClusters operator()( VPLightClusters const& ) const override;

    mutable Gaudi::Accumulators::AveragingCounter<unsigned long> m_convertedClusters{ this, "# Converted Clusters" };
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::VPLightClustersToVPMicroClustersConverter,
                           "VPLightClustersToVPMicroClustersConverter" )

LHCb::VPMicroClusters
LHCb::VPLightClustersToVPMicroClustersConverter::operator()( LHCb::VPLightClusters const& clusters ) const {
  LHCb::VPMicroClusters microClusters;

  microClusters.reserve( clusters.size() );
  for ( auto pCluster = clusters.rbegin(); pCluster != clusters.rend(); pCluster++ ) {
    VPMicroCluster cluster( pCluster->xfraction(), pCluster->yfraction(), pCluster->channelID() );
    microClusters.insert( &cluster, &cluster + 1, cluster.channelID().module() );
  }

  m_convertedClusters += clusters.size();

  return microClusters;
}
