/*****************************************************************************\
* (c) Copyright 2018 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "Event/PrDownstreamTracks.h"
#include "Event/PrLongTracks.h"
#include "Event/PrUpstreamTracks.h"
#include "Event/PrVeloTracks.h"
#include "Event/Track_v1.h"
#include "Event/Track_v3.h"

namespace LHCb::Pr::ConversionInfo {

  template <Event::Enum::Track::Type Type>
  constexpr auto stateLocations() {
    constexpr auto state_locs = Event::v3::get_state_locations<Event::v3::available_states_t<Type>>{};
    return state_locs();
  }

  struct Downstream {
    static constexpr LHCb::Event::v1::Track::Types   Type      = LHCb::Event::v1::Track::Types::Downstream;
    static constexpr LHCb::Event::v1::Track::History PrHistory = LHCb::Event::v1::Track::History::PrDownstream;
    using Ancestor1                                            = LHCb::Pr::Downstream::Tag::trackSeed;

    static constexpr auto                       StateLocations        = stateLocations<Type>();
    static constexpr bool                       AddStatesFromAncestor = false;
    static constexpr std::array<const char*, 1> AncestorLocations     = { "SeedTracksLocation" };
  };

  struct Upstream {
    static constexpr LHCb::Event::v1::Track::Types   Type             = LHCb::Event::v1::Track::Types::Upstream;
    static constexpr LHCb::Event::v1::Track::History PrHistory        = LHCb::Event::v1::Track::History::PrVeloUT;
    using Ancestor1                                                   = LHCb::Pr::Upstream::Tag::trackVP;
    static constexpr auto                       StateLocations        = stateLocations<Type>();
    static constexpr bool                       AddStatesFromAncestor = false;
    static constexpr std::array<const char*, 1> AncestorLocations     = { "VeloTracksLocation" };
  };

  struct Match {
    static constexpr LHCb::Event::v1::Track::Types   Type             = LHCb::Event::v1::Track::Types::Long;
    static constexpr LHCb::Event::v1::Track::History PrHistory        = LHCb::Event::v1::Track::History::PrMatch;
    using Ancestor1                                                   = LHCb::Pr::Long::Tag::trackVP;
    using Ancestor2                                                   = LHCb::Pr::Long::Tag::trackSeed;
    static constexpr auto                       StateLocations        = stateLocations<Type>();
    static constexpr bool                       AddStatesFromAncestor = true;
    static constexpr std::array<const char*, 2> AncestorLocations     = { "VeloTracksLocation", "SeedTracksLocation" };
  };

  struct Forward {
    static constexpr LHCb::Event::v1::Track::Types   Type      = LHCb::Event::v1::Track::Types::Long;
    static constexpr LHCb::Event::v1::Track::History PrHistory = LHCb::Event::v1::Track::History::PrForward;
    using Ancestor1                                            = LHCb::Pr::Long::Tag::trackVP;

    static constexpr auto                       StateLocations        = stateLocations<Type>();
    static constexpr bool                       AddStatesFromAncestor = false;
    static constexpr std::array<const char*, 1> AncestorLocations     = { "VeloTracksLocation" };
  };

  struct ForwardFromVeloUT {
    static constexpr LHCb::Event::v1::Track::Types   Type      = LHCb::Event::v1::Track::Types::Long;
    static constexpr LHCb::Event::v1::Track::History PrHistory = LHCb::Event::v1::Track::History::PrForward;
    using Ancestor1                                            = LHCb::Pr::Long::Tag::trackUT;

    static constexpr auto                       StateLocations        = stateLocations<Type>();
    static constexpr bool                       AddStatesFromAncestor = false;
    static constexpr std::array<const char*, 1> AncestorLocations     = { "UpstreamTracksLocation" };
  };

  struct VeloForward {
    static constexpr LHCb::Event::v1::Track::Types   Type      = LHCb::Event::v1::Track::Types::Velo;
    static constexpr LHCb::Event::v1::Track::History PrHistory = LHCb::Event::v1::Track::History::PrPixel;

    static constexpr auto                       StateLocations        = stateLocations<Type>();
    static constexpr bool                       AddStatesFromAncestor = false;
    static constexpr std::array<const char*, 0> AncestorLocations     = {};
  };

  struct VeloBackward {
    static constexpr LHCb::Event::v1::Track::Types   Type      = LHCb::Event::v1::Track::Types::VeloBackward;
    static constexpr LHCb::Event::v1::Track::History PrHistory = LHCb::Event::v1::Track::History::PrPixel;

    static constexpr auto                       StateLocations        = stateLocations<Type>();
    static constexpr bool                       AddStatesFromAncestor = false;
    static constexpr std::array<const char*, 0> AncestorLocations     = { {} };
  };

  struct Velo {
    static LHCb::Event::v1::Track::Types Type( bool backward = false ) {
      if ( backward )
        return LHCb::Event::v1::Track::Types::VeloBackward;
      else
        return LHCb::Event::v1::Track::Types::Velo;
    }
    static constexpr LHCb::Event::v1::Track::History PrHistory = LHCb::Event::v1::Track::History::PrPixel;

    static constexpr auto                       StateLocations = stateLocations<Event::Enum::Track::Type::Velo>();
    static constexpr bool                       AddStatesFromAncestor = false;
    static constexpr std::array<const char*, 0> AncestorLocations     = {};
  };

  struct Seeding {
    static constexpr LHCb::Event::v1::Track::Types   Type      = LHCb::Event::v1::Track::Types::Ttrack;
    static constexpr LHCb::Event::v1::Track::History PrHistory = LHCb::Event::v1::Track::History::PrSeeding;

    static constexpr auto                       StateLocations        = stateLocations<Type>();
    static constexpr bool                       AddStatesFromAncestor = false;
    static constexpr std::array<const char*, 0> AncestorLocations     = { {} };
  };

} // namespace LHCb::Pr::ConversionInfo
