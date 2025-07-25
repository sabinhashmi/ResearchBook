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

#include "Event/FTLiteCluster.h"
#include "LHCbAlgs/Consumer.h"

using FTLiteClusters = LHCb::FTLiteCluster::FTLiteClusters;

namespace {
  bool is_same_cluster( LHCb::FTLiteCluster const& ftlc_a, LHCb::FTLiteCluster const& ftlc_b ) {
    return ftlc_a.getLiteCluster() == ftlc_b.getLiteCluster();
  }
} // namespace

namespace LHCb {
  /**
   *  Tests to see whether all of the clusters in the FTLiteClusters container
   *  are indeed present in the other FTLiteClusters container.
   *
   *  @author Laurent Dufour
   */
  struct TestOverlappingFTLiteClusters : Algorithm::Consumer<void( FTLiteClusters const&, FTLiteClusters const& )> {

    TestOverlappingFTLiteClusters( const std::string& name, ISvcLocator* pSvcLocator )
        : Consumer( name, pSvcLocator, { KeyValue{ "FTLiteClusters_A", "" }, KeyValue{ "FTLiteClusters_B", "" } } ) {}
    void operator()( FTLiteClusters const&, FTLiteClusters const& ) const override;

    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_missed_in_B{ this, "Missing clusters in [B]" };
    mutable Gaudi::Accumulators::Counter<>                m_found_in_B{ this, "Found clusters in [B]" };
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::TestOverlappingFTLiteClusters, "TestOverlappingFTLiteClusters" )

void LHCb::TestOverlappingFTLiteClusters::operator()( FTLiteClusters const& ftlcs_a,
                                                      FTLiteClusters const& ftlcs_b ) const {
  /// Tests for the presence of the FTLiteCluster from A in the Container B
  for ( const auto& ft_lite_cluster : ftlcs_a.range() ) {
    const auto& globalQuarterIdx = ft_lite_cluster.channelID().globalQuarterIdx();

    auto it = std::find_if(
        ftlcs_b.range( globalQuarterIdx ).begin(), ftlcs_b.range( globalQuarterIdx ).end(),
        [&ft_lite_cluster]( const FTLiteCluster& ftlc_b ) { return is_same_cluster( ft_lite_cluster, ftlc_b ); } );

    if ( it == ftlcs_b.range( globalQuarterIdx ).end() ) {
      ++m_missed_in_B;
    } else {
      ++m_found_in_B;
    }
  }
}
