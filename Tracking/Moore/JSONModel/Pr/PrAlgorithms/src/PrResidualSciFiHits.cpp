/*****************************************************************************\
* (c) Copyright 2000-2024 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "Gaudi/Accumulators.h"

#include "Event/PrHits.h"
#include "Event/PrLongTracks.h"
#include "FTDAQ/FTInfo.h"
#include "LHCbAlgs/Transformer.h"

#include "boost/dynamic_bitset.hpp"

#include <array>
#include <limits>

namespace LHCb::Pr::FT {

  /**
   * @brief An algorithm that flags SciFi hits used by Long tracks and returns a SciFi hit
   * container containing residual unflagged hits.
   *
   */
  class ResidualHits
      : public Algorithm::Transformer<Hits( const EventContext&, const LHCb::Pr::Long::Tracks&, const Hits& )> {
    using Tracks = Long::Tracks;

    mutable Gaudi::Accumulators::Counter<> m_noTracks{ this, "Empty Long tracks" };

  public:
    ResidualHits( const std::string& name, ISvcLocator* pSvcLocator );

    Hits operator()( const EventContext&, const Tracks&, const Hits& ) const override;
  };

  // Declaration of the Algorithm Factory
  DECLARE_COMPONENT_WITH_ID( ResidualHits, "PrResidualSciFiHits" )

  //=============================================================================
  // Standard constructor, initializes variables
  //=============================================================================
  ResidualHits::ResidualHits( const std::string& name, ISvcLocator* pSvcLocator )
      : Transformer( name, pSvcLocator,
                     { KeyValue{ "TracksLocation", "" }, KeyValue{ "SciFiHitsLocation", PrFTInfo::SciFiHitsLocation } },
                     KeyValue{ "SciFiHitsOutput", PrFTInfo::SciFiHitsLocation } ) {}

  //=============================================================================
  // Main execution
  //=============================================================================
  Hits ResidualHits::operator()( const EventContext& evtCtx, const Tracks& tracks, const Hits& fthits ) const {

    Hits hits{ LHCb::getMemResource( evtCtx ), fthits.size() };

    if ( tracks.empty() || fthits.empty() ) {
      if ( tracks.empty() ) { ++m_noTracks; }
      // explicit copy required (to generally avoid copies)
      hits.copy_from( fthits );
      return hits;
    }

    auto used = boost::dynamic_bitset<>{ fthits.size() };

    /// mark used SciFi Hits
    for ( const auto& track : tracks.scalar() ) {
      const int nfthits = track.nFTHits().cast();
      for ( int id = 0; id != nfthits; id++ ) {
        const auto idx = track.ft_index( id ).cast();
        used[idx]      = true;
      }
    }

    for ( auto zone : hitzones ) {
      const auto [zoneBegin, zoneEnd] = fthits.getZoneIndices( zone );
      hits.setZoneIndex( zone, hits.size() );
      for ( auto iHit{ zoneBegin }; iHit < zoneEnd; ++iHit ) {
        if ( used[iHit] ) continue;
        hits.appendColumn( fthits.planeCode( iHit ), fthits.x( iHit ), fthits.hotHitInfo( iHit ),
                           fthits.coldHitInfo( iHit ) );
      }
      hits.appendColumn( std::numeric_limits<uint8_t>::max(), 1.e9f, {}, {} );
    }

    hits.setZoneIndex( LHCb::Detector::FT::nZonesTotal, hits.getZoneIndex( xu[0] ) );
    // so when asking for lastZone, lastZone+2 gives the very end of the container
    hits.setZoneIndex( LHCb::Detector::FT::nZonesTotal + 1, hits.size() );
    // avoid FPEs
    for ( unsigned i{ 0 }; i < SIMDWrapper::best::types::size; ++i ) {
      hits.appendColumn( std::numeric_limits<uint8_t>::max(), 1.e9f, {}, {} );
    }

    return hits;
  }

} // namespace LHCb::Pr::FT
