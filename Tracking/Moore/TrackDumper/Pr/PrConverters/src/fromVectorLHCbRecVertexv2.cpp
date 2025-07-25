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
#include "Event/RecVertex.h"
#include "Event/RecVertex_v2.h"
#include "Event/Track.h"
#include "Event/Track_v2.h"
#include "Kernel/EventLocalAllocator.h"
#include "LHCbAlgs/Transformer.h"
#include <assert.h>

/**
 *  Small helper to convert std::vector<LHCb::RecVertex> to LHCb::RecVertices
 */
namespace LHCb::Converters::RecVertex::v2 {

  template <typename InputType>
  using Transformer = Algorithm::Transformer<LHCb::RecVertices( const InputType&, const LHCb::Track::Range& )>;

  template <typename InputType>
  struct fromv2RecVertex : Transformer<InputType> {
    using KeyValue = typename Transformer<InputType>::KeyValue;

    fromv2RecVertex( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer<InputType>{ name,
                                  pSvcLocator,
                                  { KeyValue{ "InputVerticesName", "" }, KeyValue{ "InputTracksName", "" } },
                                  KeyValue{ "OutputVerticesName", "" } } {}

    /// The main function, converts the vertex and puts it into a keyed container
    LHCb::RecVertices operator()( const InputType& vertices, const LHCb::Track::Range& keyed_tracks ) const override {
      LHCb::RecVertices converted_vertices;
      for ( const auto& vertex : vertices ) {
        auto converted_vertex = std::make_unique<LHCb::RecVertex>( vertex.position() );
        converted_vertex->setTechnique( LHCb::RecVertex::RecVertexType::Primary );
        converted_vertex->setChi2( vertex.chi2() );
        converted_vertex->setNDoF( vertex.nDoF() );
        converted_vertex->setCovMatrix( vertex.covMatrix() );
        // The following relies on the Velo tracks being created with a key in PrPixelTracking.
        for ( const auto& weightedTrack : vertex.tracks() ) {
          if ( weightedTrack.track == nullptr ) continue;
          auto track_in_keyed_container =
              std::find_if( std::begin( keyed_tracks ), std::end( keyed_tracks ), [&weightedTrack]( const auto& t ) {
                return ( ( t->nLHCbIDs() == weightedTrack.track->nLHCbIDs() ) &&
                         ( t->containsLhcbIDs( weightedTrack.track->lhcbIDs() ) ) );
              } );
          if ( track_in_keyed_container != std::end( keyed_tracks ) ) {
            converted_vertex->addToTracks( *track_in_keyed_container, weightedTrack.weight );
          } else {
            throw GaudiException( "Could not find Track corresponding to RecVertex v2. "
                                  "Check the data passed to InputTracksName.",
                                  this->name(), StatusCode::FAILURE );
          }
        }
        converted_vertices.add( converted_vertex.release() );
      }
      return converted_vertices;
    }
  };

  using fromLHCbRecVertices = fromv2RecVertex<LHCb::Event::v2::RecVertices>;
  DECLARE_COMPONENT_WITH_ID( fromLHCbRecVertices, "LHCb__Converters__RecVertex__v2__fromVectorLHCbRecVertices" )

  using fromVectorLHCbRecVertex = fromv2RecVertex<LHCb::Event::v2::RecVertices>;
  DECLARE_COMPONENT_WITH_ID( fromVectorLHCbRecVertex, "LHCb__Converters__RecVertex__v2__fromVectorLHCbRecVertex" )
} // namespace LHCb::Converters::RecVertex::v2

namespace LHCb::Converters::RecVertices {

  struct LHCbRecVerticesToVectorV2RecVertex
      : public Algorithm::Transformer<LHCb::Event::v2::RecVertices( LHCb::RecVertices const&,
                                                                    std::vector<LHCb::Event::v2::Track> const& )> {
    LHCbRecVerticesToVectorV2RecVertex( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator, { KeyValue{ "InputVertices", "" }, KeyValue{ "InputTracks", "" } },
                       KeyValue{ "OutputVertices", "" } ) {}
    /// The main function, converts the vertex and puts it into a keyed container
    LHCb::Event::v2::RecVertices operator()( LHCb::RecVertices const&                   vertices,
                                             std::vector<LHCb::Event::v2::Track> const& tracks ) const override {
      LHCb::Event::v2::RecVertices converted_vertices;
      for ( const auto* vertex : vertices ) {
        if ( not vertex ) continue;
        converted_vertices.emplace_back( LHCb::Event::v2::RecVertex( vertex->position(), vertex->covMatrix(),
                                                                     { vertex->chi2PerDoF(), vertex->nDoF() } ) );
        // Add tracks from keyed container.
        // The following relies on the Velo tracks being created with a key in PrPixelTracking.
        for ( const auto& weightedTrack : vertex->tracksWithWeights() ) {
          if ( not weightedTrack.first ) continue;
          auto new_track_it =
              std::find_if( std::begin( tracks ), std::end( tracks ), [&weightedTrack]( const auto& t ) {
                return ( ( t.nLHCbIDs() == weightedTrack.first->nLHCbIDs() ) &&
                         ( t.containsLhcbIDs( weightedTrack.first->lhcbIDs() ) ) );
              } );
          assert( new_track_it != std::end( tracks ) );
          converted_vertices.back().addToTracks( &*new_track_it, weightedTrack.second );
        }
      }
      return converted_vertices;
    }
  };
  DECLARE_COMPONENT( LHCbRecVerticesToVectorV2RecVertex )
} // namespace LHCb::Converters::RecVertices
