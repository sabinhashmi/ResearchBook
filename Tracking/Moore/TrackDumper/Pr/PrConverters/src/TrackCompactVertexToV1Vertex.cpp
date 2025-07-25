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
#include "Event/Track_v3.h"
#include "LHCbAlgs/Transformer.h"
#include "PrKernel/PrSelection.h"
#include "SelKernel/TrackZips.h"
#include "TrackKernel/TrackCompactVertex.h"
#include <assert.h>

namespace {
  /** @brief Create and fill vertex information from a RecVertex from a TrackCompactVertex.
   */
  LHCb::RecVertex create_vertex( const LHCb::TrackKernel::TrackCompactVertex<2, double>& input ) {
    auto output = LHCb::RecVertex{ input.position() };
    // output->setTechnique( static_cast<LHCb::RecVertex::RecVertexType>( input.technique() ) );
    output.setChi2( input.chi2() );
    output.setNDoF( input.nDoF() );
    output.setCovMatrix( input.posCovMatrix() );
    return output;
  }

  /** @brief Find a track based on a set of LHCbIDs and add it to the vertex.
   *
   * Asserts that a track based on the LHCbIDs can be found.
   *
   * @param source_track_ids LHCbIDs used to find the track in converted_tracks
   * @param converted_tracks Container in which to search for the track to be added to vertex
   * @param vertex Vertex to which the found track will be added
   */
  void add_track_from_ids_to_vertex( const std::vector<LHCb::LHCbID>& source_track_ids,
                                     const std::vector<LHCb::Track>& converted_tracks, LHCb::RecVertex& vertex ) {
    auto track_in_converted_container =
        std::find_if( std::begin( converted_tracks ), std::end( converted_tracks ), [&]( const auto& t ) {
          return ( ( t.nLHCbIDs() == source_track_ids.size() ) && ( t.containsLhcbIDs( source_track_ids ) ) );
        } );
    assert( track_in_converted_container != std::end( converted_tracks ) );
    vertex.addToTracks( &*track_in_converted_container, 1 ); // TODO 1 -> real weight
  }
} // namespace

namespace LHCb::Converters::TrackCompactVertex {

  /** @brief Convert a vector of two-body TrackCompactVertex to a vector of RecVertex.
   *
   * The LHCb::TrackKernel::TrackCompactVertex is backed by a
   * LHCb::Pr::Selection of LHCb::Event::v2::Track objects.
   */
  struct VectorOf2Trackv2CompactVertexToVectorOfRecVertex
      : public Algorithm::Transformer<std::vector<LHCb::RecVertex>(
            std::vector<LHCb::TrackKernel::TrackCompactVertex<2, double>> const&,
            const ::Pr::Selection<LHCb::Event::v2::Track>&, const std::vector<LHCb::Track>& )> {
    VectorOf2Trackv2CompactVertexToVectorOfRecVertex( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator,
                       { KeyValue{ "InputVertices", "" }, KeyValue{ "TracksInVertices", "" },
                         KeyValue{ "ConvertedTracks", "" } },
                       KeyValue{ "OutputVertices", "" } ) {}
    std::vector<LHCb::RecVertex>
    operator()( const std::vector<LHCb::TrackKernel::TrackCompactVertex<2, double>>& vertices,
                const ::Pr::Selection<LHCb::Event::v2::Track>&                       tracks,
                const std::vector<LHCb::Track>&                                      conv_tracks ) const override {
      std::vector<LHCb::RecVertex> converted_vertices;
      for ( const auto& vertex : vertices ) {
        auto converted_vertex = create_vertex( vertex );
        ;
        for ( int i = 0; i < 2; ++i ) {
          auto const& ids = tracks[vertex.child_relations()[i].index()].lhcbIDs();
          add_track_from_ids_to_vertex( ids, conv_tracks, converted_vertex );
        }
        converted_vertices.push_back( std::move( converted_vertex ) );
      }
      return converted_vertices;
    }
  };
  DECLARE_COMPONENT( VectorOf2Trackv2CompactVertexToVectorOfRecVertex )

  /** @brief Convert a vector of two-body TrackCompactVertex to a vector of RecVertex.
   *
   * The The LHCb::TrackKernel::TrackCompactVertex is backed by tracks of type
   * LHCb::Event::v3::<something>.
   *
   * @tparam VertexTrackType The type of the tracks held by the vertex.
   */
  template <typename VertexTrackType>
  using Transformer = Algorithm::Transformer<std::vector<LHCb::RecVertex>(
      std::vector<LHCb::TrackKernel::TrackCompactVertex<2, double>,
                  LHCb::Allocators::EventLocal<LHCb::TrackKernel::TrackCompactVertex<2, double>>> const&,
      const VertexTrackType&, const std::vector<LHCb::Track>& )>;
  template <typename VertexTrackType>
  struct VectorOf2TrackPrFittedCompactVertexToVectorOfRecVertex : public Transformer<VertexTrackType> {

    using KeyValue = typename Transformer<VertexTrackType>::KeyValue;

    VectorOf2TrackPrFittedCompactVertexToVectorOfRecVertex( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer<VertexTrackType>( name, pSvcLocator,
                                        { KeyValue{ "InputVertices", "" }, KeyValue{ "TracksInVertices", "" },
                                          KeyValue{ "ConvertedTracks", "" } },
                                        KeyValue{ "OutputVertices", "" } ) {}
    std::vector<LHCb::RecVertex> operator()(
        const std::vector<LHCb::TrackKernel::TrackCompactVertex<2, double>,
                          LHCb::Allocators::EventLocal<LHCb::TrackKernel::TrackCompactVertex<2, double>>>& vertices,
        const VertexTrackType& tracks_zip, const std::vector<LHCb::Track>& conv_tracks ) const override {
      std::vector<LHCb::RecVertex> converted_vertices;
      const auto&                  tracks = tracks_zip.template get<LHCb::Event::v3::Tracks>();

      const auto fittedtracks = tracks.scalar();
      for ( const auto& vertex : vertices ) {
        auto converted_vertex = create_vertex( vertex );
        ;
        for ( int i = 0; i < 2; ++i ) {
          auto const idx = vertex.child_relations()[i].index();
          auto       ids = fittedtracks[idx].lhcbIDs();
          // The LHCb::Event::v1::Track::containsLhcbIDs method implicitly
          // assumes that the input IDs are sorted; ordering is not guaranteed
          // by the fitted tracks so we must do that here
          std::sort( std::begin( ids ), std::end( ids ) );
          add_track_from_ids_to_vertex( ids, conv_tracks, converted_vertex );
        }
        converted_vertices.push_back( std::move( converted_vertex ) );
      }
      return converted_vertices;
    }
  };
  DECLARE_COMPONENT_WITH_ID(
      VectorOf2TrackPrFittedCompactVertexToVectorOfRecVertex<LHCb::Event::v3::TracksWithMuonID>,
      "LHCb__Converters__TrackCompactVertex__VectorOf2TrackPrFittedWithMuonIDCompactVertexToVectorOfRecVertex" )
  DECLARE_COMPONENT_WITH_ID(
      VectorOf2TrackPrFittedCompactVertexToVectorOfRecVertex<LHCb::Event::v3::TracksWithPVs>,
      "LHCb__Converters__TrackCompactVertex__VectorOf2TrackPrFittedWithPVsCompactVertexToVectorOfRecVertex" )
} // namespace LHCb::Converters::TrackCompactVertex
