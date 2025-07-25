/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
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
#include <algorithm>
#include <vector>

namespace LHCb::Converters::RecVertex::v1 {

  /** @brief Convert Event::v2::RecVertices to a vector of LHCb::RecVertex
   *
   * Part of vertex conversion is conversion of track references to
   * corresponding tracks of an appropriate format.  In addition to the input
   * container of Event::v2::RecVertex instances to be converted, an input
   * vector of Event::v1::Track instances is required as the target of the
   * converted track references.  For each Event::v2::Track referenced by
   * a Event::v2::RecVertex to be converted, the corresponding Event::v1::Track
   * is found by matching lists of LHCb IDs and added to the converted
   * LHCb::RecVertex with the same weight.
   */
  struct fromRecVertexv2RecVertexv1 : public Algorithm::Transformer<std::vector<LHCb::RecVertex>(
                                          const Event::v2::RecVertices&, std::vector<Event::v1::Track> const& )> {

    fromRecVertexv2RecVertexv1( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer{ name,
                       pSvcLocator,
                       { KeyValue{ "InputVerticesName", "" }, KeyValue{ "InputTracksName", "" } },
                       KeyValue{ "OutputVerticesName", "" } } {}

    std::vector<LHCb::RecVertex> operator()( const Event::v2::RecVertices&        vertexes,
                                             std::vector<Event::v1::Track> const& tracks ) const override {
      std::vector<LHCb::RecVertex> converted_vertexes;
      for ( const auto& vertex : vertexes ) {
        auto& converted_vertex = converted_vertexes.emplace_back( LHCb::RecVertex( vertex.position() ) );
        converted_vertex.setChi2( vertex.chi2() );
        converted_vertex.setNDoF( vertex.nDoF() );
        converted_vertex.setCovMatrix( vertex.covMatrix() );

        for ( const auto& weightedTrack : vertex.tracks() ) {
          if ( weightedTrack.track == nullptr ) continue;
          auto matched_track_it =
              std::find_if( std::begin( tracks ), std::end( tracks ), [&weightedTrack]( const auto& t ) {
                return ( std::equal( t.lhcbIDs().begin(), t.lhcbIDs().end(), weightedTrack.track->lhcbIDs().begin(),
                                     weightedTrack.track->lhcbIDs().end() ) );
              } );

          if ( matched_track_it != std::end( tracks ) ) {
            converted_vertex.addToTracks( &*matched_track_it, weightedTrack.weight );
          } else {
            throw GaudiException( "Could not find Track corresponding to RecVertex v2. "
                                  "Check the data passed to InputTracksName.",
                                  this->name(), StatusCode::FAILURE );
          }
        }
      }
      m_nbVtxsCounter += converted_vertexes.size();
      return converted_vertexes;
    }

  private:
    // A counter for the converted vertexes
    mutable Gaudi::Accumulators::SummingCounter<> m_nbVtxsCounter{ this, "Nb of converted Vertices" };
  };

  DECLARE_COMPONENT( fromRecVertexv2RecVertexv1 )
} // namespace LHCb::Converters::RecVertex::v1
