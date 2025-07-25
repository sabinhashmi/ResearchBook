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
#include "Event/Track.h"
#include "LHCbAlgs/Transformer.h"
#include <cassert>

/** @class fromVectorLHCbRecVertex fromVectorLHCbRecVertex.h
 *
 *  Small helper to convert std::vector<LHCb::RecVertex> to LHCb::RecVertices
 *
 */

namespace LHCb::Converters::RecVertex::v1 {
  struct fromVectorLHCbRecVertex
      : public Algorithm::Transformer<RecVertices( const std::vector<LHCb::RecVertex>&, const Tracks& )> {
    fromVectorLHCbRecVertex( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator, { KeyValue{ "InputVerticesName", "" }, KeyValue{ "InputTracksName", "" } },
                       KeyValue{ "OutputVerticesName", "" } ) {}
    /// The main function, converts the vertex and puts it into a keyed container
    RecVertices operator()( const std::vector<LHCb::RecVertex>& vertices, const Tracks& keyed_tracks ) const override {
      RecVertices converted_vertices;
      for ( const auto& vertex : vertices ) {
        auto converted_vertex = std::make_unique<LHCb::RecVertex>( vertex );
        converted_vertex->clearTracks();
        // Add tracks from keyed container.
        // The following relies on the Velo tracks being created with a key in PrPixelTracking.
        for ( const auto& vertex_track : vertex.tracksWithWeights() ) {
          auto track_in_keyed_container =
              std::find_if( std::begin( keyed_tracks ), std::end( keyed_tracks ), [&vertex_track]( const auto& t ) {
                return ( ( t->nLHCbIDs() == vertex_track.first->nLHCbIDs() ) &&
                         ( t->nCommonLhcbIDs( *vertex_track.first ) == t->nLHCbIDs() ) );
              } );

          // Track* track_in_keyed_container =
          // dynamic_cast<Track*>(keyed_tracks.containedObject(vertex_track.first->key()));
          // assert((track_in_keyed_container->nLHCbIDs()==vertex_track.first->nLHCbIDs()) &&
          //       (track_in_keyed_container->nCommonLhcbIDs(*vertex_track.first)==track_in_keyed_container->nLHCbIDs()));
          assert( track_in_keyed_container != std::end( keyed_tracks ) );
          converted_vertex->addToTracks( *track_in_keyed_container, vertex_track.second );
        }
        converted_vertices.add( converted_vertex.release() );
      }
      return converted_vertices;
    }
  };
  DECLARE_COMPONENT( fromVectorLHCbRecVertex )
} // namespace LHCb::Converters::RecVertex::v1
