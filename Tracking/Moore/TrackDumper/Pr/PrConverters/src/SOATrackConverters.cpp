/*****************************************************************************\
 * (c) Copyright 2000-2020 CERN for the benefit of the LHCb Collaboration      *
 *                                                                             *
 * This software is distributed under the terms of the GNU General Public      *
 * Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "Event/PrLongTracks.h"
#include "Event/PrUpstreamTracks.h"
#include "Event/PrVeloTracks.h"
#include "Event/RelationTable.h"
#include "Event/SOATrackConversion.h"
#include "Event/State.h"
#include "Event/StateParameters.h"
#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "Event/UniqueIDGenerator.h"
#include "LHCbAlgs/Transformer.h"
#include "SelKernel/TrackZips.h"
#include "SelKernel/Utilities.h"
#include <vector>

/**
 * Converter between vector<LHCb::Event::v1::Track> and LHCb::Event::v3::Tracks ( SoA PoD )
 * including relations (v1 track pointer for v3 track index)
 *
 * @author Miguel Ramos Pernas
 */

namespace {
  using namespace LHCb::Event;
  using dType     = SIMDWrapper::scalar::types;
  using I         = dType::int_v;
  using F         = dType::float_v;
  using OutTracks = LHCb::Event::v3::Tracks;
  template <typename TrackContainer>
  using output_t       = std::tuple<OutTracks, V3ToV1Mapping<TrackContainer>>;
  namespace conversion = LHCb::Event::conversion;

  template <typename TrackContainer, typename TrackObj_t>
  int get_index_v1( TrackObj_t const& track ) {
    if constexpr ( std::is_same<LHCb::Event::v1::Tracks, TrackContainer>::value ) {
      return track->key();
    } else {
      return track->index();
    }
  }

} // namespace

namespace LHCb::Converters::v2::Event {
  template <typename TrackContainer>
  using Transformer = typename Algorithm::MultiTransformer<output_t<TrackContainer>( const TrackContainer&,
                                                                                     const LHCb::UniqueIDGenerator& )>;

  template <typename TrackContainer>
  class fromTrack : public Transformer<TrackContainer> {

  public:
    using Output   = output_t<TrackContainer>;
    using KeyValue = typename Transformer<TrackContainer>::KeyValue;

    fromTrack( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer<TrackContainer>{
              name,
              pSvcLocator,
              { KeyValue{ "InputTracks", "" },
                KeyValue{ "InputUniqueIDGenerator", LHCb::UniqueIDGeneratorLocation::Default } },
              { KeyValue{ "OutputTracks", "" }, KeyValue{ "Relations", "" } } } {}

    Output operator()( const TrackContainer&          fitted_tracks,
                       const LHCb::UniqueIDGenerator& unique_id_gen ) const override {
      auto zn = Zipping::generateZipIdentifier();
      // check if there are tracks
      if ( fitted_tracks.empty() ) {
        ++m_emptyTracks;
        // assume that track was fitted
        auto output = Output{ OutTracks( m_only_type, Enum::Track::FitHistory::PrKalmanFilter,
                                         m_only_type == Enum::Track::Type::VeloBackward, unique_id_gen, zn ),
                              V3ToV1Mapping<TrackContainer>( &fitted_tracks ) };
        return output;
      }

      const auto& firstTrack = Sel::Utils::deref_if_ptr( *fitted_tracks.begin() );
      const auto  outputTrackType =
          m_only_type != LHCb::Event::v3::TrackType::Unknown ? m_only_type.value() : firstTrack.type();
      // find the first track with selected track type and use its fit history
      const auto firstOfType      = std::find_if( fitted_tracks.begin(), fitted_tracks.end(), [&]( const auto& track ) {
        return Sel::Utils::deref_if_ptr( track ).type() == outputTrackType;
      } );
      const auto outputFitHistory = firstOfType != fitted_tracks.end()
                                        ? Sel::Utils::deref_if_ptr( *firstOfType ).fitHistory()
                                        : Enum::Track::FitHistory::PrKalmanFilter;

      auto output            = Output{ OutTracks( outputTrackType, outputFitHistory,
                                                  outputTrackType == Enum::Track::Type::VeloBackward, unique_id_gen, zn ),
                            V3ToV1Mapping<TrackContainer>( &fitted_tracks ) };
      auto& [out, relations] = output;

      out.reserve( fitted_tracks.size() );
      relations.reserve( fitted_tracks.size() );

      for ( auto const& rtrack : fitted_tracks ) {
        if ( !conversion::ref_is_valid( rtrack ) ) continue;
        if ( m_only_type.value() != LHCb::Event::v3::TrackType::Unknown &&
             m_only_type.value() != Sel::Utils::deref_if_ptr( rtrack ).type() )
          continue;

        auto outTrack = out.template emplace_back<SIMDWrapper::InstructionSet::Scalar>();
        if ( outTrack.fitHistory() != Sel::Utils::deref_if_ptr( rtrack ).fitHistory() ) {
          throw GaudiException( "Expected fit history " + toString( outTrack.fitHistory() ) + " but got " +
                                    toString( Sel::Utils::deref_if_ptr( rtrack ).fitHistory() ),
                                this->name(), StatusCode::FAILURE );
        }
        conversion::Status status, result = conversion::convert_track( out.type(), outTrack, rtrack, unique_id_gen );
        do {
          std::tie( result, status ) = conversion::process_result( result );

          if ( status == conversion::DifferentType ) {
            ++m_different_types;
          } else if ( status == conversion::InvalidStates ) {
            ++m_invalid_states;
          }
        } while ( status != conversion::Success );

        relations.add( outTrack.indices().cast(), get_index_v1<TrackContainer>( rtrack ) );
      }
      m_nbTracksCounter += out.size();
      return output;
    }

  private:
    Gaudi::Property<LHCb::Event::v3::TrackType> m_only_type{
        this, "RestrictToType", LHCb::Event::v3::TrackType::Unknown,
        "If set, filter the input tracks and only write those of the given type. Otherwise the full set is processed, "
        "and the type of the container is determined from the first track to convert (a warning is displayed if tracks "
        "of different types are consequently found)" };

    mutable Gaudi::Accumulators::SummingCounter<>         m_nbTracksCounter{ this, "Nb of Produced Tracks" };
    mutable Gaudi::Accumulators::SummingCounter<>         m_emptyTracks{ this, "Nb of Events without Tracks" };
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_different_types{
        this, "Container is being filled with tracks of different types" };
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_invalid_states{
        this, "Invalid states detected in track to convert" };
  };

  DECLARE_COMPONENT_WITH_ID( fromTrack<LHCb::Event::v1::Tracks>, "LHCb__Converters__Track__SOA__fromV1Track" )
  DECLARE_COMPONENT_WITH_ID( fromTrack<SharedObjectsContainer<LHCb::Event::v1::Track>>,
                             "LHCb__Converters__Track__SOA__fromSharedV1Track" )

} // namespace LHCb::Converters::v2::Event
