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
  LHCb::Event::v1::Track ConvertTrack( LHCb::Event::v2::Track const& track ) {
    auto outTr = LHCb::Event::v1::Track{};
    outTr.setChi2PerDoF( track.chi2PerDoF() );
    outTr.setNDoF( track.nDoF() );
    outTr.setLikelihood( 999.9f );
    outTr.setGhostProbability( 999.9f );
    outTr.setFlags( track.flags() );
    outTr.addToLhcbIDs( track.lhcbIDs() );

    // copy the states
    for ( auto const& state : track.states() ) { outTr.addToStates( state ); }

    return outTr;
  }
} // namespace
namespace LHCb::Converters::Track::v1 {

  struct fromV2TrackV1Track
      : public Algorithm::ScalarTransformer<fromV2TrackV1Track,
                                            Event::v1::Tracks( std::vector<Event::v2::Track> const& )> {

    fromV2TrackV1Track( std::string const& name, ISvcLocator* pSvcLocator )
        : ScalarTransformer( name, pSvcLocator, KeyValue{ "InputTracksName", "" },
                             KeyValue{ "OutputTracksName", "" } ) {}

    using ScalarTransformer::operator();

    /// The main function, converts the track
    Event::v1::Track* operator()( Event::v2::Track const& track ) const {
      return new Event::v1::Track{ ConvertTrack( track ) };
    }
  };
  DECLARE_COMPONENT( fromV2TrackV1Track )

  struct fromV2TrackV1TrackVector
      : public Algorithm::ScalarTransformer<fromV2TrackV1TrackVector,
                                            std::vector<Event::v1::Track>( std::vector<Event::v2::Track> const& )> {

    fromV2TrackV1TrackVector( std::string const& name, ISvcLocator* pSvcLocator )
        : ScalarTransformer( name, pSvcLocator, KeyValue{ "InputTracksName", "" },
                             KeyValue{ "OutputTracksName", "" } ) {}

    using ScalarTransformer::operator();

    /// The main function, converts the track
    Event::v1::Track operator()( Event::v2::Track const& track ) const { return ConvertTrack( track ); }
  };
  DECLARE_COMPONENT( fromV2TrackV1TrackVector )

  struct fromV2SelectionV1TrackVector
      : public Algorithm::Transformer<std::vector<Event::v1::Track>( Pr::Selection<Event::v2::Track> const& )> {

    fromV2SelectionV1TrackVector( std::string const& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator, KeyValue{ "InputTracksName", "" }, KeyValue{ "OutputTracksName", "" } ) {}

    /// The main function, converts the track
    std::vector<Event::v1::Track> operator()( Pr::Selection<Event::v2::Track> const& tracks ) const override {

      std::vector<Event::v1::Track> output;
      output.reserve( tracks.size() );

      std::transform( tracks.begin(), tracks.end(), std::back_inserter( output ), ConvertTrack );

      return output;
    }
  };
  DECLARE_COMPONENT( fromV2SelectionV1TrackVector )

} // namespace LHCb::Converters::Track::v1
