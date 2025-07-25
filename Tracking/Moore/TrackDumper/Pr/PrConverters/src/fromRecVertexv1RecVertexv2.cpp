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

namespace LHCb::Converters::RecVertex::v2 {
  /** @brief Convert vector of LHCb::RecVertex to Event::v2::RecVertices
   *
   * Part of vertex conversion is conversion of track references to
   * corresponding tracks of an appropriate format.  In addition to the input
   * container of LHCb::RecVertex instances to be converted, an input
   * vector of Event::v2::Track instances is required as the target of the
   * converted track references.  For each Event::v1::Track referenced by
   * a LHCb::RecVertex to be converted, the corresponding Event::v2::Track
   * is found by matching lists of LHCb IDs and added to the converted
   * Event::v2::RecVertex with the same weight.
   */
  struct fromRecVertexv1RecVertexv2 : public Algorithm::Transformer<Event::v2::RecVertices(
                                          const std::vector<LHCb::RecVertex>&, std::vector<Event::v2::Track> const& )> {

    fromRecVertexv1RecVertexv2( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer{ name,
                       pSvcLocator,
                       { KeyValue{ "InputVerticesName", "" }, KeyValue{ "InputTracksName", "" } },
                       KeyValue{ "OutputVerticesName", "" } } {}

    Event::v2::RecVertices operator()( const std::vector<LHCb::RecVertex>&  vertexes,
                                       std::vector<Event::v2::Track> const& tracks ) const override {
      Event::v2::RecVertices converted_vertexes;
      for ( const auto& vertex : vertexes ) {
        auto& converted_vertex = converted_vertexes.emplace_back(
            Event::v2::RecVertex( vertex.position(), vertex.covMatrix(), { vertex.chi2(), vertex.nDoF() } ) );

        for ( const auto& weightedTrack : vertex.tracksWithWeights() ) {
          if ( not weightedTrack.first ) continue;
          auto matched_track_it =
              std::find_if( std::begin( tracks ), std::end( tracks ), [&weightedTrack]( const auto& t ) {
                return ( std::equal( t.lhcbIDs().begin(), t.lhcbIDs().end(), weightedTrack.first->lhcbIDs().begin(),
                                     weightedTrack.first->lhcbIDs().end() ) );
              } );

          if ( matched_track_it != std::end( tracks ) ) {
            converted_vertex.addToTracks( &*matched_track_it, weightedTrack.second );
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

  DECLARE_COMPONENT( fromRecVertexv1RecVertexv2 )
} // namespace LHCb::Converters::RecVertex::v2
