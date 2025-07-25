/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "Event/Particle_v2.h"
#include "Event/RecVertex.h"
#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "Event/ZipUtils.h"
#include "LHCbAlgs/Transformer.h"
#include "SelKernel/TrackZips.h"
#include <cassert>

namespace {
  /** @brief Find a track based on a set of LHCbIDs and add it to the vertex.
   *
   * Asserts that a track based on the LHCbIDs can be found.
   *
   * @param source_track_ids LHCbIDs used to find the track in converted_tracks
   * @param converted_tracks Container in which to search for the track to be added to vertex
   * @param vertex Vertex to which the found track will be added
   *
   * @todo This was copied from TrackCompactVertexToV1Vertex.cpp, the
   *       duplication should be removed in future.
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

  /** @brief Create a RecVertex from a row of an LHCb::Event::Composites container
   *  @todo  Provide a nicer alias for generating the first argument type name.
   */
  LHCb::RecVertex convert_vertex(
      LHCb::Event::Composites::template proxy_type<SIMDWrapper::Scalar, LHCb::Pr::ProxyBehaviour::Contiguous,
                                                   LHCb::Event::Composites const>
                                     composite,
      LHCb::Event::v3::Tracks const& tracks, std::vector<LHCb::Track> const& conv_tracks ) {
    LHCb::RecVertex output{ Gaudi::XYZPoint{ LHCb::Utils::as_arithmetic( composite.x() ),
                                             LHCb::Utils::as_arithmetic( composite.y() ),
                                             LHCb::Utils::as_arithmetic( composite.z() ) } };
    output.setChi2AndDoF( LHCb::Utils::as_arithmetic( composite.chi2() ),
                          LHCb::Utils::as_arithmetic( composite.nDoF() ) );
    {
      Gaudi::SymMatrix3x3 covMatrix{};
      for ( auto i = 0; i < 3; ++i ) {
        for ( auto j = i; j < 3; ++j ) {
          covMatrix( i, j ) = LHCb::Utils::as_arithmetic( composite.posCovElement( i, j ) );
        }
      }
      output.setCovMatrix( covMatrix );
    }
    for ( auto i = 0u; i < composite.numChildren(); ++i ) {
      // Get the zip family identifier for the i-th child of the composite
      Zipping::ZipFamilyNumber child_container_family{
          LHCb::Utils::as_arithmetic( composite.childRelationFamily( i ) ) };
      // Check it matches the track container that we were given
      if ( child_container_family != tracks.zipIdentifier() ) {
        std::ostringstream oss;
        oss << "Mismatch in zip family identifier, needed " << child_container_family
            << " but the provided track container has " << tracks.zipIdentifier();
        throw GaudiException{ oss.str(), "CompositesToV1Vertex::{anon}::convert_vertex", StatusCode::FAILURE };
      }
      auto const fittedtracks = tracks.scalar();
      auto const childIdx     = LHCb::Utils::as_arithmetic( composite.childRelationIndex( i ) );
      auto       ids          = fittedtracks[childIdx].lhcbIDs();
      // The LHCb::Event::v1::Track::containsLhcbIDs method implicitly
      // assumes that the input IDs are sorted; ordering is not guaranteed
      // by the fitted tracks so we must do that here
      using std::begin;
      using std::end;
      std::sort( begin( ids ), end( ids ) );
      add_track_from_ids_to_vertex( ids, conv_tracks, output );
    }

    return output;
  }
} // namespace

namespace LHCb::Converters::Composites {
  /** @brief Convert a LHCb::Event::Composites container to a vector of RecVertex.
   *
   * LHCb::Event::Composites only holds the indices and zip family IDs of child
   * particles, so the containers of children must be explicitly passed in to
   * the converter. Here we assume that there is only a single child container,
   * and that it is of type `tracks_zip_t`.
   *
   * @tparam tracks_zip_t The type of the tracks held by the vertex, this must
   *                      be a zip including LHCb::Event::v3::Tracks
   *
   * @todo This algorithm shouldn't need to be a template (for the current two
   *       instantiations), as the configuration could do some more traversal
   *       of the data dependency tree and find the
   *       LHCb::Event::v3::Tracks container that was originally used
   *       to make the zip (tracks_zip_t) that we actually pass in.
   */
  template <typename tracks_zip_t>
  using ToVectorOfRecVertex_base_t = Algorithm::Transformer<std::vector<RecVertex>(
      Event::Composites const&, tracks_zip_t const&, std::vector<Track> const& )>;
  template <typename tracks_zip_t>
  struct ToVectorOfRecVertex : public ToVectorOfRecVertex_base_t<tracks_zip_t> {
    using base_t   = ToVectorOfRecVertex_base_t<tracks_zip_t>;
    using KeyValue = typename base_t::KeyValue;

    ToVectorOfRecVertex( const std::string& name, ISvcLocator* pSvcLocator )
        : base_t{ name,
                  pSvcLocator,
                  { KeyValue{ "InputComposites", "" }, KeyValue{ "TracksInVertices", "" },
                    KeyValue{ "ConvertedTracks", "" } },
                  KeyValue{ "OutputVertices", "" } } {}
    std::vector<RecVertex> operator()( Event::Composites const& composites, tracks_zip_t const& tracks_zip,
                                       std::vector<Track> const& conv_tracks ) const override {
      // Create the output container and reserve capacity
      std::vector<RecVertex> converted_vertices;
      converted_vertices.reserve( composites.size() );

      // Extract the fitted tracks from the zip -- see @todo above
      auto const& tracks = tracks_zip.template get<Event::v3::Tracks>();
      for ( auto const composite : composites.scalar() ) {
        converted_vertices.emplace_back( convert_vertex( composite, tracks, conv_tracks ) );
      }
      return converted_vertices;
    }
  };
  DECLARE_COMPONENT_WITH_ID( ToVectorOfRecVertex<Event::v3::TracksWithMuonID>,
                             "LHCb__Converters__Composites__TracksWithMuonIDToVectorOfRecVertex" )
  DECLARE_COMPONENT_WITH_ID( ToVectorOfRecVertex<Event::v3::TracksWithPVs>,
                             "LHCb__Converters__Composites__TracksWithPVsToVectorOfRecVertex" )
} // namespace LHCb::Converters::Composites
