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

#include "Event/PrHits.h"
#include "Event/UTLiteCluster.h"
#include "LHCbAlgs/Consumer.h"
#include "PrKernel/UTHit.h"
#include "PrKernel/UTHitHandler.h"

namespace LHCb {
  /**
   *  Tests to see whether all of the clusters in the PrUTHits container
   *  (i.e. the SoA container)
   *  are indeed present in the other SoA container.
   *
   *  When ran in debug mode, provides additional information cluster by
   *  cluster.
   *
   *  Main use-case for this algorithm is debugging and testing of the
   *  converters and persistency.
   *
   *  @author Laurent Dufour
   */
  struct TestOverlappingUTHits : Algorithm::Consumer<void( ::LHCb::Pr::UT::Hits const&, ::LHCb::Pr::UT::Hits const& )> {

    TestOverlappingUTHits( const std::string& name, ISvcLocator* pSvcLocator )
        : Consumer( name, pSvcLocator, { KeyValue{ "UTHits_A", "" }, KeyValue{ "UTHits_B", "" } } ) {}
    void operator()( ::LHCb::Pr::UT::Hits const&, ::LHCb::Pr::UT::Hits const& ) const override;

    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_missed_in_B{ this, "Missing clusters in [B]" };
    mutable Gaudi::Accumulators::Counter<>                m_found_in_B{ this, "Found clusters in [B]" };

    template <typename T>
    bool is_same_uthit( T const& uth_a, T const& uth_b ) const {
      LHCb::Detector::UT::ChannelID channelID_A( uth_a.template get<LHCb::Pr::UT::UTHitsTag::channelID>().cast() );
      LHCb::Detector::UT::ChannelID channelID_B( uth_b.template get<LHCb::Pr::UT::UTHitsTag::channelID>().cast() );

      const double       fracStrip_A     = uth_a.template get<LHCb::Pr::UT::UTHitsTag::fracStrip>().cast();
      const unsigned int clusterSize_A   = uth_a.template get<LHCb::Pr::UT::UTHitsTag::clusterSize>().cast();
      const auto         clusterCharge_A = uth_a.template get<LHCb::Pr::UT::UTHitsTag::clusterCharge>().cast();

      const double       fracStrip_B     = uth_b.template get<LHCb::Pr::UT::UTHitsTag::fracStrip>().cast();
      const unsigned int clusterSize_B   = uth_b.template get<LHCb::Pr::UT::UTHitsTag::clusterSize>().cast();
      const auto         clusterCharge_B = uth_b.template get<LHCb::Pr::UT::UTHitsTag::clusterCharge>().cast();

      if ( channelID_A != channelID_B ) // the most obvious one
        return false;

      bool good_cluster = true;

      // let's check things that should match, on the lowest level
      if ( clusterSize_A != clusterSize_B || std::min( 1023, clusterCharge_A ) != std::min( 1023, clusterCharge_B ) ) {
        if ( this->msgLevel( MSG::VERBOSE ) ) {
          this->verbose() << "Mismatch: [size] (A) " << clusterSize_A;
          this->verbose() << " (B) " << clusterSize_B << endmsg;

          this->verbose() << "Mismatch: [clusterCharge] (A) " << clusterCharge_A;
          this->verbose() << " (B) " << clusterCharge_B << endmsg;
          this->verbose() << "Mismatch happening for channelID " << channelID_A << endmsg;
        }

        good_cluster = false;
      }

      if ( fabs( fracStrip_A - fracStrip_B ) > .55 / 255 ) // tolerance due to limited precision
      {
        if ( this->msgLevel( MSG::VERBOSE ) ) {
          this->verbose() << "Mismatch: [fracStrip] (A) " << fracStrip_A;
          this->verbose() << " (B) " << fracStrip_B << endmsg;
        }

        good_cluster = false;
      }

      const auto weight_A = uth_a.template get<LHCb::Pr::UT::UTHitsTag::weight>().cast();
      const auto weight_B = uth_b.template get<LHCb::Pr::UT::UTHitsTag::weight>().cast();

      const auto xAtYEq0_A = uth_a.template get<LHCb::Pr::UT::UTHitsTag::xAtYEq0>().cast();
      const auto xAtYEq0_B = uth_b.template get<LHCb::Pr::UT::UTHitsTag::xAtYEq0>().cast();

      const auto zAtYEq0_A = uth_a.template get<LHCb::Pr::UT::UTHitsTag::zAtYEq0>().cast();
      const auto zAtYEq0_B = uth_b.template get<LHCb::Pr::UT::UTHitsTag::zAtYEq0>().cast();

      const auto yBegin_A = uth_a.template get<LHCb::Pr::UT::UTHitsTag::yBegin>().cast();
      const auto yBegin_B = uth_b.template get<LHCb::Pr::UT::UTHitsTag::yBegin>().cast();

      const auto yEnd_A = uth_a.template get<LHCb::Pr::UT::UTHitsTag::yEnd>().cast();
      const auto yEnd_B = uth_b.template get<LHCb::Pr::UT::UTHitsTag::yEnd>().cast();

      const auto cos_A = uth_a.template get<LHCb::Pr::UT::UTHitsTag::cos>().cast();
      const auto cos_B = uth_b.template get<LHCb::Pr::UT::UTHitsTag::cos>().cast();

      const auto dxDy_A = uth_a.template get<LHCb::Pr::UT::UTHitsTag::dxDy>().cast();
      const auto dxDy_B = uth_b.template get<LHCb::Pr::UT::UTHitsTag::dxDy>().cast();

      // now derived quantities
      // there are these (seemingly arbitrary) tolerances on the differences
      // due to floating point precision
      // precision of 2 microns on x
      if ( ( weight_A != weight_B ) || ( fabs( xAtYEq0_A - xAtYEq0_B ) > 2 * Gaudi::Units::micrometer ) ||
           ( zAtYEq0_A != zAtYEq0_B ) || ( fabs( yBegin_A - yBegin_B ) > 0.0001 ) ||
           ( fabs( yEnd_A - yEnd_B ) > 1.5 * Gaudi::Units::micrometer ) ||
           ( fabs( cos_A - cos_B ) > 0.001 ) || // can depend on fracstrip; looser tolerance
           ( fabs( dxDy_A - dxDy_B ) > 0.001 ) ) {
        if ( this->msgLevel( MSG::VERBOSE ) ) {
          this->verbose() << "Mismatch in derived quantities: " << endmsg;
          this->verbose() << "Fracstrip: " << fracStrip_A << " vs " << fracStrip_B << "("
                          << ( fracStrip_A - fracStrip_B ) << ")" << endmsg;
          this->verbose() << "weight: " << weight_A << " vs " << weight_B << "(" << ( weight_A - weight_B ) << ")"
                          << endmsg;
          this->verbose() << "xAtYEq0: " << xAtYEq0_A << " vs " << xAtYEq0_B << "(" << ( xAtYEq0_A - xAtYEq0_B ) << ")"
                          << endmsg;
          this->verbose() << "zAtYEq0: " << zAtYEq0_A << " vs " << zAtYEq0_B << "(" << ( zAtYEq0_A - zAtYEq0_B ) << ")"
                          << endmsg;
          this->verbose() << "yBegin: " << yBegin_A << " vs " << yBegin_B << "(" << ( yBegin_A - yBegin_B ) << ")"
                          << endmsg;
          this->verbose() << "yEnd: " << yEnd_A << " vs " << yEnd_B << "(" << ( yEnd_A - yEnd_B ) << ")" << endmsg;
          this->verbose() << "cos: " << cos_A << " vs " << cos_B << "(" << ( cos_A - cos_B ) << ")" << endmsg;
          this->verbose() << "dxDy: " << dxDy_A << " vs " << dxDy_B << "(" << ( dxDy_A - dxDy_B ) << ")" << endmsg;
        }

        good_cluster = false;
      }

      return good_cluster;
    }
  };

} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::TestOverlappingUTHits, "TestOverlappingUTHits" )

void LHCb::TestOverlappingUTHits::operator()( ::LHCb::Pr::UT::Hits const& utlcs_a,
                                              ::LHCb::Pr::UT::Hits const& utlcs_b ) const {
  for ( const auto& ut_hit_a : utlcs_a.scalar() ) {
    LHCb::Detector::UT::ChannelID channelID_A( ut_hit_a.get<LHCb::Pr::UT::UTHitsTag::channelID>().cast() );

    // auto [index_a, index_b] = utlcs_b.indices( channelID_A.sectorFullID() );

    auto it = std::find_if( utlcs_b.scalar().begin(), utlcs_b.scalar().end(),
                            [&]( const auto& utlc_b ) { return is_same_uthit( ut_hit_a, utlc_b ); } );
    if ( it == utlcs_b.scalar().end() ) {
      ++m_missed_in_B;
    } else {
      ++m_found_in_B;
    }
  }
}
