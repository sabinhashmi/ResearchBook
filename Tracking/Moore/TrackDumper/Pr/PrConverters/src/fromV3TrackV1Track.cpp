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
#include "Event/GhostProbability.h"
#include "Event/PartialChiSquareds.h"
#include "Event/PrKalmanFitResult.h"
#include "Event/SOATrackConversion.h"
#include "Event/StateParameters.h"
#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "GaudiKernel/GaudiException.h"
#include "GaudiKernel/StdArrayAsProperty.h"
#include "LHCbAlgs/Transformer.h"
#include "SelKernel/TrackZips.h"
#include "TrackInterfaces/IGhostProbability.h"
#include "TrackInterfaces/ITrackAddClusters.h"
#include <vector>

/**
 * Converter between LHCb::Event::v3::Tracks ( SoA PoD ) and vector<Track_v1>
 *
 */

namespace {
  using dType = SIMDWrapper::scalar::types;
  using I     = dType::int_v;
  using F     = dType::float_v;

  namespace conversion = LHCb::Event::conversion;

  template <typename Type, typename... Types>
  inline constexpr bool has_input_type = ( std::is_same_v<Type, Types> || ... );

  template <typename T>
  auto get_input_name() {
    if constexpr ( std::is_same_v<T, LHCb::Event::v3::Tracks> )
      return std::string( "InputTracks" );
    else if constexpr ( std::is_same_v<T, LHCb::Event::v3::TracksWithMuonID> )
      return std::string( "InputTracks" );
    else if constexpr ( std::is_same_v<T, LHCb::Event::v3::TracksWithPVs> )
      return std::string( "InputTracks" );
    else if constexpr ( std::is_same_v<T, LHCb::Event::v3::Track::PartialChiSquareds> )
      return std::string( "InputPartialChi2s" );
    else if constexpr ( std::is_same_v<T, LHCb::Event::v3::GhostProbabilities> )
      return std::string( "InputGhostProbs" );
    else if constexpr ( std::is_same_v<T, std::vector<LHCb::PrKalmanFitResult>> )
      return std::string( "InputTrackFitResults" );
    else
      throw GaudiException( "Input type is not supported", "fromV3TrackV1Track", StatusCode::FAILURE );
  }

  /// Update a single state
  template <LHCb::Event::v3::Tracks::StateLocation L, typename TrackProxy>
  void update_state( LHCb::Event::v1::Track& new_track, TrackProxy const& inTrack ) {

    auto        s = inTrack.state( L );
    LHCb::State state;
    state.setState( s.x().cast(), s.y().cast(), s.z().cast(), s.tx().cast(), s.ty().cast(), s.qOverP().cast() );
    state.setLocation( L );
    state.setCovariance( LHCb::LinAlg::convert<double>( inTrack.covariance( L ) ) );
    new_track.addToStates( state );
  }

  template <typename PartialChiSquaredsProxy>
  void add_partial_chi2s( LHCb::Event::v1::Track& new_track, PartialChiSquaredsProxy const& in_partial_chi2 ) {

    using out_track = LHCb::Event::v1::Track;

    switch ( new_track.type() ) {
    case out_track::Types::Long:
      new_track.addInfo( out_track::AdditionalInfo::FitTChi2, in_partial_chi2.FitTChi2().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitTNDoF, in_partial_chi2.FitTNDoF().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitVeloChi2, in_partial_chi2.FitVeloChi2().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitVeloNDoF, in_partial_chi2.FitVeloNDoF().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitMatchChi2, in_partial_chi2.FitMatchChi2().cast() );
      new_track.addInfo( out_track::AdditionalInfo::NUTOutliers, in_partial_chi2.NUTOutliers().cast() );
      break;
    case out_track::Types::Downstream:
      new_track.addInfo( out_track::AdditionalInfo::FitTChi2, in_partial_chi2.FitTChi2().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitTNDoF, in_partial_chi2.FitTNDoF().cast() );
      new_track.addInfo( out_track::AdditionalInfo::NUTOutliers, in_partial_chi2.NUTOutliers().cast() );
      break;
    case out_track::Types::Velo:
      new_track.addInfo( out_track::AdditionalInfo::FitVeloChi2, in_partial_chi2.FitVeloChi2().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitVeloNDoF, in_partial_chi2.FitVeloNDoF().cast() );
      break;
    case out_track::Types::VeloBackward:
      new_track.addInfo( out_track::AdditionalInfo::FitVeloChi2, in_partial_chi2.FitVeloChi2().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitVeloNDoF, in_partial_chi2.FitVeloNDoF().cast() );
      break;
    case out_track::Types::Ttrack:
      new_track.addInfo( out_track::AdditionalInfo::FitTChi2, in_partial_chi2.FitTChi2().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitTNDoF, in_partial_chi2.FitTNDoF().cast() );
      break;
    case out_track::Types::Upstream:
      new_track.addInfo( out_track::AdditionalInfo::FitVeloChi2, in_partial_chi2.FitVeloChi2().cast() );
      new_track.addInfo( out_track::AdditionalInfo::FitVeloNDoF, in_partial_chi2.FitVeloNDoF().cast() );
      new_track.addInfo( out_track::AdditionalInfo::NUTOutliers, in_partial_chi2.NUTOutliers().cast() );
      break;
    default:
      throw GaudiException( "PartialChiSquareds is empty for this track type", "fromV3TrackV1Track",
                            StatusCode::FAILURE );
    }
  }

  /// Actual implementation of the "update_states" function
  template <typename TrackProxy, LHCb::Event::v3::Tracks::StateLocation... L>
  static void update_states_impl( LHCb::Event::v1::Track& new_track, TrackProxy const& inTrack,
                                  LHCb::Event::v3::state_collection<L...> ) {
    ( update_state<L>( new_track, inTrack ), ... );
  }

  /// Update all the states of a track
  template <typename TrackProxy>
  void update_states( LHCb::Event::v1::Track& new_track, TrackProxy const& inTrack, LHCb::Event::v3::TrackType type ) {

    using trtype_t = LHCb::Event::v3::TrackType;
    switch ( type ) {
    case LHCb::Event::v3::TrackType::Long:
      switch ( inTrack.fitHistory() ) {
      case LHCb::Event::Enum::Track::FitHistory::Unknown:
        return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::Long>{} );
      case LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter:
      case LHCb::Event::Enum::Track::FitHistory::TrackMasterFitter:
        return update_states_impl(
            new_track, inTrack,
            LHCb::Event::v3::available_states_t<trtype_t::Long,
                                                LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>{} );
      case LHCb::Event::Enum::Track::FitHistory::VeloKalman:
        return update_states_impl(
            new_track, inTrack,
            LHCb::Event::v3::available_states_t<trtype_t::Long, LHCb::Event::Enum::Track::FitHistory::VeloKalman>{} );
      case LHCb::Event::Enum::Track::FitHistory::Last:
        throw GaudiException( "Last is not a valid FitHistory.", "SOATrackConversion", StatusCode::FAILURE );
      }
      __builtin_unreachable();
    case LHCb::Event::v3::TrackType::Downstream:
      switch ( inTrack.fitHistory() ) {
      case LHCb::Event::Enum::Track::FitHistory::Unknown:
        return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::Downstream>{} );
      case LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter:
      case LHCb::Event::Enum::Track::FitHistory::TrackMasterFitter:
        return update_states_impl(
            new_track, inTrack,
            LHCb::Event::v3::available_states_t<trtype_t::Downstream,
                                                LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>{} );
      case LHCb::Event::Enum::Track::FitHistory::VeloKalman:
      case LHCb::Event::Enum::Track::FitHistory::Last:
        throw GaudiException( "Not a valid FitHistory.", "SOATrackConversion", StatusCode::FAILURE );
      }
      __builtin_unreachable();
    case LHCb::Event::v3::TrackType::Velo:
      switch ( inTrack.fitHistory() ) {
      case LHCb::Event::Enum::Track::FitHistory::Unknown:
        return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::Velo>{} );
      case LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter:
      case LHCb::Event::Enum::Track::FitHistory::TrackMasterFitter:
        return update_states_impl(
            new_track, inTrack,
            LHCb::Event::v3::available_states_t<trtype_t::Velo,
                                                LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>{} );
      case LHCb::Event::Enum::Track::FitHistory::VeloKalman:
      case LHCb::Event::Enum::Track::FitHistory::Last:
        throw GaudiException( "Not a valid FitHistory.", "SOATrackConversion", StatusCode::FAILURE );
      }
      __builtin_unreachable();
    case LHCb::Event::v3::TrackType::VeloBackward:
      switch ( inTrack.fitHistory() ) {
      case LHCb::Event::Enum::Track::FitHistory::Unknown:
        return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::VeloBackward>{} );
      case LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter:
      case LHCb::Event::Enum::Track::FitHistory::TrackMasterFitter:
        return update_states_impl(
            new_track, inTrack,
            LHCb::Event::v3::available_states_t<trtype_t::VeloBackward,
                                                LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>{} );
      case LHCb::Event::Enum::Track::FitHistory::VeloKalman:
      case LHCb::Event::Enum::Track::FitHistory::Last:
        throw GaudiException( "Not a valid FitHistory.", "SOATrackConversion", StatusCode::FAILURE );
      }
      __builtin_unreachable();
    case LHCb::Event::v3::TrackType::Ttrack:
      switch ( inTrack.fitHistory() ) {
      case LHCb::Event::Enum::Track::FitHistory::Unknown:
        return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::Ttrack>{} );
      case LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter:
      case LHCb::Event::Enum::Track::FitHistory::TrackMasterFitter:
        return update_states_impl(
            new_track, inTrack,
            LHCb::Event::v3::available_states_t<trtype_t::Ttrack,
                                                LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>{} );
      case LHCb::Event::Enum::Track::FitHistory::VeloKalman:
      case LHCb::Event::Enum::Track::FitHistory::Last:
        throw GaudiException( "Not a valid FitHistory.", "SOATrackConversion", StatusCode::FAILURE );
      }
      __builtin_unreachable();
    case LHCb::Event::v3::TrackType::Upstream:
      switch ( inTrack.fitHistory() ) {
      case LHCb::Event::Enum::Track::FitHistory::Unknown:
        return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::Upstream>{} );
      case LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter:
      case LHCb::Event::Enum::Track::FitHistory::TrackMasterFitter:
        return update_states_impl(
            new_track, inTrack,
            LHCb::Event::v3::available_states_t<trtype_t::Upstream,
                                                LHCb::Event::Enum::Track::FitHistory::PrKalmanFilter>{} );
      case LHCb::Event::Enum::Track::FitHistory::VeloKalman:
      case LHCb::Event::Enum::Track::FitHistory::Last:
        throw GaudiException( "Not a valid FitHistory.", "SOATrackConversion", StatusCode::FAILURE );
      }
      __builtin_unreachable();
    case LHCb::Event::v3::TrackType::Muon:
      return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::Muon>{} );
    case LHCb::Event::v3::TrackType::UT:
      return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::UT>{} );
    case LHCb::Event::v3::TrackType::Unknown:
      return update_states_impl( new_track, inTrack, LHCb::Event::v3::available_states_t<trtype_t::Unknown>{} );
    default:
      throw GaudiException( "unknown v3 track type", "fromV3TrackV1Track", StatusCode::FAILURE );
    }
    __builtin_unreachable();
  }

  template <typename TrackProxy>
  void convert_track( LHCb::Event::v1::Track& new_track, TrackProxy const& inTrack ) {

    new_track.setHistory( inTrack.history().cast() );
    new_track.setFitHistory( inTrack.fitHistory() );
    new_track.setPatRecStatus( LHCb::Event::v1::Track::PatRecStatus::PatRecIDs );
    new_track.setFitStatus( inTrack.fitHistory() != LHCb::Event::Enum::Track::FitHistory::Unknown
                                ? LHCb::Event::v1::Track::FitStatus::Fitted
                                : LHCb::Event::v1::Track::FitStatus::Unknown );

    new_track.setLhcbIDs( inTrack.lhcbIDs() );
    new_track.setChi2AndDoF( inTrack.chi2().cast(), inTrack.nDoF().cast() );

    update_states( new_track, inTrack, inTrack.type() );
  }
} // namespace

namespace LHCb::Converters::Track::v1 {

  template <typename... V3InputTypes>
  class fromV3TrackV1Track : public LHCb::Algorithm::Transformer<LHCb::Event::v1::Tracks( V3InputTypes const&... )> {

  public:
    using base_class = LHCb::Algorithm::Transformer<LHCb::Event::v1::Tracks( V3InputTypes const&... )>;
    using KeyValue   = typename base_class::KeyValue;

    fromV3TrackV1Track( const std::string& name, ISvcLocator* pSvcLocator )
        : base_class( name, pSvcLocator, { KeyValue( get_input_name<V3InputTypes>(), "" )... },
                      KeyValue{ "OutputTracks", "" } ) {}

    LHCb::Event::v1::Tracks operator()( V3InputTypes const&... inputs ) const override {

      auto out = LHCb::Event::v1::Tracks{};

      const auto  input_tuple = std::forward_as_tuple( inputs... );
      const auto& in_tracks   = ( [&]() -> auto& {
        if constexpr ( std::is_same_v<decltype( std::get<0>( input_tuple ) ), LHCb::Event::v3::Tracks const&> ) {
          return std::get<LHCb::Event::v3::Tracks const&>( input_tuple );
        } else { // if it's a zip
          return std::get<0>( input_tuple ).template get<LHCb::Event::v3::Tracks>();
        }
      } )();

      constexpr bool has_partial_chisquareds =
          has_input_type<LHCb::Event::v3::Track::PartialChiSquareds, V3InputTypes...>;
      constexpr bool has_ghost_probabilities = has_input_type<LHCb::Event::v3::GhostProbabilities, V3InputTypes...>;
      constexpr bool has_fit_results         = has_input_type<std::vector<LHCb::PrKalmanFitResult>, V3InputTypes...>;

      const auto& tracks_with_extra_infos = [&] {
        if constexpr ( has_ghost_probabilities ) {
          const auto& extra_infos = std::get<LHCb::Event::v3::GhostProbabilities const&>( input_tuple );
          assert( extra_infos.size() == in_tracks.size() );
          return LHCb::Event::make_zip<SIMDWrapper::InstructionSet::Scalar>( in_tracks, extra_infos );
        } else if constexpr ( has_partial_chisquareds ) {
          const auto& extra_infos = std::get<LHCb::Event::v3::Track::PartialChiSquareds const&>( input_tuple );
          assert( extra_infos.size() == in_tracks.size() );
          return LHCb::Event::make_zip<SIMDWrapper::InstructionSet::Scalar>( in_tracks, extra_infos );
        } else {
          return LHCb::Event::make_zip<SIMDWrapper::InstructionSet::Scalar>( in_tracks );
        }
      }();

      out.reserve( in_tracks.size() );

      int counter = 0;
      for ( auto const track_with_extra_info : tracks_with_extra_infos ) {
        auto new_track = new LHCb::Event::v1::Track{};
        if ( in_tracks.backward() )
          new_track->setType( LHCb::Track::Types::VeloBackward );
        else
          new_track->setType( track_with_extra_info.type() );

        convert_track( *new_track, track_with_extra_info );

        auto sc_clusters_on_track = m_clusterAdder->fillClustersFromLHCbIDs( *new_track );
        if ( !sc_clusters_on_track.isSuccess() ) ++m_clustersOnTrackFail;

        if constexpr ( has_partial_chisquareds ) { add_partial_chi2s( *new_track, track_with_extra_info ); }

        if constexpr ( has_ghost_probabilities ) {
          new_track->setGhostProbability(
              track_with_extra_info.template get<LHCb::Event::v3::GhostProbabilityTag::GhostProbability>().cast() );
        } else if constexpr ( !has_partial_chisquareds ) {
          new_track->setGhostProbability( std::numeric_limits<double>::max() );
          ++m_no_ghostProb;
        }

        if constexpr ( has_fit_results ) {
          const auto& fit_results = std::get<std::vector<LHCb::PrKalmanFitResult> const&>( input_tuple );
          assert( fit_results.size() == in_tracks.size() );
          new_track->setFitResult( new LHCb::PrKalmanFitResult{ std::move( fit_results[counter] ) } );
        }
        out.insert( new_track );
        ++counter;
      }

      // add ghost probability in batched evaluation
      if constexpr ( !has_ghost_probabilities && has_partial_chisquareds ) {
        for ( auto gt : m_ghostTools ) gt->execute( out ).ignore();
      }

      m_nbTracksCounter += out.size();
      return out;
    }

  private:
    ToolHandle<ITrackAddClusters>      m_clusterAdder{ this, "TrackAddClusterTool", "AddClustersToTrackTool" };
    ToolHandleArray<IGhostProbability> m_ghostTools{
        this, "GhostProbTools", {}, "List of ghost probability tools (per track type)" };
    mutable Gaudi::Accumulators::SummingCounter<>         m_nbTracksCounter{ this, "Nb of Produced Tracks" };
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_no_ghostProb{ this,
                                                                          "Track ghostProb set to default value" };
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_clustersOnTrackFail{
        this, "Not all LHCbIDs were found as clusters." };
  };

  template <typename output_t, typename... V3InputTypes>
  class fromV3TrackV1TrackVector : public LHCb::Algorithm::Transformer<output_t( V3InputTypes const&... )> {

  public:
    using base_class = LHCb::Algorithm::Transformer<output_t( V3InputTypes const&... )>;
    using KeyValue   = typename base_class::KeyValue;

    fromV3TrackV1TrackVector( const std::string& name, ISvcLocator* pSvcLocator )
        : base_class( name, pSvcLocator, { KeyValue( get_input_name<V3InputTypes>(), "" )... },
                      KeyValue{ "OutputTracks", "" } ) {}

    output_t operator()( V3InputTypes const&... inputs ) const override {

      auto out = output_t{};

      const auto  input_tuple = std::forward_as_tuple( inputs... );
      const auto& in_tracks   = ( [&]() -> auto& {
        if constexpr ( std::is_same_v<decltype( std::get<0>( input_tuple ) ), LHCb::Event::v3::Tracks const&> ) {
          return std::get<LHCb::Event::v3::Tracks const&>( input_tuple );
        } else { // if it's a zip
          return std::get<0>( input_tuple ).template get<LHCb::Event::v3::Tracks>();
        }
      } )();

      const auto& tracks_with_extra_infos = [&] {
        return LHCb::Event::make_zip<SIMDWrapper::InstructionSet::Scalar>( in_tracks );
      }();

      out.reserve( in_tracks.size() );

      for ( auto const track_with_extra_info : tracks_with_extra_infos ) {
        auto new_track = new LHCb::Event::v1::Track{};
        if ( in_tracks.backward() )
          new_track->setType( LHCb::Track::Types::VeloBackward );
        else
          new_track->setType( track_with_extra_info.type() );

        convert_track( *new_track, track_with_extra_info );

        auto sc_clusters_on_track = m_clusterAdder->fillClustersFromLHCbIDs( *new_track );
        if ( !sc_clusters_on_track.isSuccess() ) ++m_clustersOnTrackFail;

        new_track->setGhostProbability( std::numeric_limits<double>::max() );

        if constexpr ( std::is_same_v<LHCb::Event::v1::Tracks, output_t> ) {
          out.insert( new_track );
        } else {
          out.emplace_back( *new_track );
        }
      }

      m_nbTracksCounter += out.size();
      return out;
    }

  private:
    mutable Gaudi::Accumulators::SummingCounter<> m_nbTracksCounter{ this, "Nb of Produced Tracks" };
    ToolHandle<ITrackAddClusters>                 m_clusterAdder{ this, "TrackAddClusterTool", "" };

    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_clustersOnTrackFail{
        this, "Not all LHCbIDs were found as clusters." };
  };

  using ConvFromV3TrackV1Track = fromV3TrackV1TrackVector<LHCb::Event::v1::Tracks, LHCb::Event::v3::Tracks>;
  using ConvFromV3TrackExtraV1Track =
      fromV3TrackV1Track<LHCb::Event::v3::Tracks, LHCb::Event::v3::Track::PartialChiSquareds>;
  using ConvFromV3TrackFullV1Track =
      fromV3TrackV1Track<LHCb::Event::v3::Tracks, LHCb::Event::v3::Track::PartialChiSquareds,
                         std::vector<LHCb::PrKalmanFitResult>>;
  using ConvFromV3TrackWithGhostProbV1Track =
      fromV3TrackV1Track<LHCb::Event::v3::Tracks, LHCb::Event::v3::GhostProbabilities>;
  using ConvFromV3TrackFullWithGhostProbV1Track =
      fromV3TrackV1Track<LHCb::Event::v3::Tracks, LHCb::Event::v3::GhostProbabilities,
                         std::vector<LHCb::PrKalmanFitResult>>;
  using ConvFromV3TrackWithMuonIDV1Track =
      fromV3TrackV1TrackVector<LHCb::Event::v1::Tracks, LHCb::Event::v3::TracksWithMuonID>;

  DECLARE_COMPONENT_WITH_ID( ConvFromV3TrackV1Track, "fromV3TrackV1Track" )
  DECLARE_COMPONENT_WITH_ID( ConvFromV3TrackExtraV1Track, "fromV3TrackExtraV1Track" )
  DECLARE_COMPONENT_WITH_ID( ConvFromV3TrackFullV1Track, "fromV3TrackFullV1Track" )
  DECLARE_COMPONENT_WITH_ID( ConvFromV3TrackWithGhostProbV1Track, "fromV3TrackWithGhostProbV1Track" )
  DECLARE_COMPONENT_WITH_ID( ConvFromV3TrackFullWithGhostProbV1Track, "fromV3TrackFullWithGhostProbV1Track" )
  DECLARE_COMPONENT_WITH_ID( ConvFromV3TrackWithMuonIDV1Track, "fromV3TrackWithMuonIDV1Track" )

  using ConvFromV3TrackWithMuonIDV1TrackVector =
      fromV3TrackV1TrackVector<std::vector<LHCb::Event::v1::Track>, LHCb::Event::v3::TracksWithMuonID>;
  using ConvFromV3TrackWithPVsV1TrackVector =
      fromV3TrackV1TrackVector<std::vector<LHCb::Event::v1::Track>, LHCb::Event::v3::TracksWithPVs>;
  DECLARE_COMPONENT_WITH_ID( ConvFromV3TrackWithMuonIDV1TrackVector, "fromV3TrackWithMuonIDV1TrackVector" )
  DECLARE_COMPONENT_WITH_ID( ConvFromV3TrackWithPVsV1TrackVector, "fromV3TrackWithPVsV1TrackVector" )

} // namespace LHCb::Converters::Track::v1
