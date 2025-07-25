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
#include "VPDet/DeVP.h"

#include "Event/VPLightCluster.h"
#include "Event/VPMicroCluster.h"
#include "LHCbAlgs/Transformer.h"

#include "PrKernel/VeloPixelInfo.h"

namespace LHCb {

  /**
   *  Transform the "VPMicroClusters" to a set of VPLiteClusters.
   *
   *  Depends on the geometry by the filling of the cluster's position information
   *
   *  @author Laurent Dufour
   */
  struct VPMicroClustersToVPLightClustersConverter
      : Algorithm::Transformer<VPLightClusters( VPMicroClusters const&, DeVP const& ),
                               Algorithm::Traits::usesConditions<DeVP>> {

    VPMicroClustersToVPLightClustersConverter( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator,
                       { KeyValue{ "InputClusters", LHCb::VPClusterLocation::Micro },
                         KeyValue{ "DeVPLocation", LHCb::Det::VP::det_path } },
                       { KeyValue{ "OutputClusters", "" } } ) {}
    LHCb::VPLightClusters operator()( VPMicroClusters const&, DeVP const& ) const override;

    mutable Gaudi::Accumulators::AveragingCounter<unsigned long> m_convertedClusters{ this, "# Converted Clusters" };
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::VPMicroClustersToVPLightClustersConverter,
                           "VPMicroClustersToVPLightClustersConverter" )

LHCb::VPLightClusters LHCb::VPMicroClustersToVPLightClustersConverter::operator()( LHCb::VPMicroClusters const& dst,
                                                                                   DeVP const& devp ) const {
  LHCb::VPLightClusters clusters;
  clusters.reserve( dst.size() );

  for ( const auto& vp_micro_cluster : dst.range() ) {
    const auto pos = vp_micro_cluster.globalPosition( devp );
    clusters.emplace_back( vp_micro_cluster.xfraction(), vp_micro_cluster.yfraction(), pos.x(), pos.y(), pos.z(),
                           vp_micro_cluster.channelID() );
  }

  sortClusterContainer( clusters );

  m_convertedClusters += clusters.size();

  return clusters;
}
