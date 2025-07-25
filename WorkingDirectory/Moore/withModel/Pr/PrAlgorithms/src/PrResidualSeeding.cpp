/*****************************************************************************\
* (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
// Include files
#include "Event/PrDownstreamTracks.h"
#include "Event/PrLongTracks.h"
#include "Event/Track.h"
#include "Gaudi/Accumulators.h"
#include "LHCbAlgs/Transformer.h"
#include "LHCbMath/SIMDWrapper.h"

#include "boost/dynamic_bitset.hpp"
#include <vector>

//-----------------------------------------------------------------------------
// class : PrResidualSeeding
// Store residual Seeding tracks after other Algorithms, e.g. PrMatchNN used
//
// 2020-04-20: Peilian Li
//
//-----------------------------------------------------------------------------
using Tracks     = LHCb::Pr::Seeding::Tracks;
using TracksLong = LHCb::Pr::Long::Tracks;
using TracksDown = LHCb::Pr::Downstream::Tracks;

template <typename TrackType>
class PrResidualSeeding : public LHCb::Algorithm::Transformer<Tracks( const TrackType&, const Tracks& )> {

  using base_class_t = LHCb::Algorithm::Transformer<Tracks( const TrackType&, const Tracks& )>;
  using base_class_t::debug;
  using base_class_t::error;
  using base_class_t::info;
  using base_class_t::inputLocation;
  using base_class_t::msgLevel;

public:
  Tracks operator()( const TrackType&, const Tracks& ) const override;

  // Declaration of the Algorithm Factory

  //=============================================================================
  // Standard constructor, initializes variables
  //=============================================================================
  PrResidualSeeding( const std::string& name, ISvcLocator* pSvcLocator )
      : base_class_t( name, pSvcLocator,
                      { typename base_class_t::KeyValue{ "MatchTracksLocation", "" },
                        typename base_class_t::KeyValue{ "SeedTracksLocation", "" } },
                      typename base_class_t::KeyValue{ "SeedTracksOutput", "" } ) {}
};
DECLARE_COMPONENT_WITH_ID( PrResidualSeeding<TracksLong>, "PrResidualSeedingLong" )

//=============================================================================
// Main execution
//=============================================================================
template <typename TrackType>
Tracks PrResidualSeeding<TrackType>::operator()( const TrackType& matchtracks, const Tracks& seedtracks ) const {
  using simd = SIMDWrapper::scalar::types;

  Tracks tmptracks{};
  tmptracks.reserve( seedtracks.size() );

  if ( seedtracks.empty() ) {
    if ( msgLevel( MSG::DEBUG ) )
      debug() << "Seed Track container '" << this->template inputLocation<Tracks>() << "' is empty" << endmsg;
    return tmptracks;
  }

  boost::dynamic_bitset<> used{ seedtracks.size(), false };

  /// mark used SciFi Hits
  for ( const auto& track : matchtracks.scalar() ) {
    const auto trackseed = track.trackSeed().cast();
    used[trackseed]      = true;
  }
  for ( auto const& track : seedtracks.scalar() ) {
    auto iTrack = track.offset();
    if ( used[iTrack] ) continue;
    tmptracks.copy_back<simd>( seedtracks, iTrack );
  }

  return tmptracks;
}
