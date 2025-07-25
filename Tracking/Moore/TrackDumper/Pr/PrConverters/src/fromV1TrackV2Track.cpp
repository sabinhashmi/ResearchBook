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
#include "Event/Track.h"
#include "Event/Track_v2.h"
#include "LHCbAlgs/ScalarTransformer.h"
#include "LHCbAlgs/Transformer.h"
#include "PrKernel/PrSelection.h"

namespace {
  /** @brief Convert a v1::Track object to a v2::Track object.
   */
  LHCb::Event::v2::Track ConvertTrack( LHCb::Event::v1::Track const& track ) {
    auto outTr = LHCb::Event::v2::Track{};
    outTr.setChi2PerDoF( { track.chi2(), track.nDoF() } );
    outTr.setFlags( track.flags() );
    for ( auto const& id : track.lhcbIDs() ) { outTr.addToLhcbIDs( id ); }
    for ( auto const state : track.states() ) { outTr.addToStates( *state ); }
    return outTr;
  }
} // namespace

namespace LHCb::Converters::Track::v2 {
  /** @brief Convert a KeyedContainer of v1::Track objects to a vector of v2::Track objects.
   *
   * Example usage:
   * @code {.py}
   * from Configurables import LHCb__Converters__Track__v2__fromV1TrackV2Track as fromV1TrackV2Track
   *
   * converter = fromV1TrackV2Track(
   *     InputTracksName='Rec/Track_v2/Best',
   *     OutputTracksName='Rec/Track_v1/Best'
   * )
   * # Then add `converter` to a dataflow
   * @endcode
   *
   * @see LHCb::Converters::Track::v1::fromV2TrackV1Track Similar algorithm for converting in the opposite direction
   */
  struct fromV1TrackV2Track : public Algorithm::ScalarTransformer<fromV1TrackV2Track, std::vector<Event::v2::Track>(
                                                                                          Event::v1::Tracks const& )> {

    fromV1TrackV2Track( std::string const& name, ISvcLocator* pSvcLocator )
        : ScalarTransformer( name, pSvcLocator, { "InputTracksName", "" }, { "OutputTracksName", "" } ) {}

    using ScalarTransformer::operator();

    Event::v2::Track operator()( Event::v1::Track const& track ) const { return ConvertTrack( track ); }
  };
  DECLARE_COMPONENT( fromV1TrackV2Track )
} // namespace LHCb::Converters::Track::v2
