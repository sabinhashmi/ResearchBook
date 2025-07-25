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
#include "Event/PrTracksTag.h"
#include "LHCbAlgs/Transformer.h"
#include "PrConvertersInfo.h"
#include "TrackInterfaces/ITrackAddClusters.h"

/** @class fromPrTracksV1Track.cpp fromPrTracksV1TrackMerger.cpp
 *  @brief Converts PrTracks to LHCb::Event::v1::Tracks (keyed container)
 *
 *  @author Michel De Cian
 *  @date 2021-06-10 First version
 */

namespace {

  struct StateErrors {
    double stateErrorX2;
    double stateErrorY2;
    double stateErrorTX2;
    double stateErrorTY2;
    double stateErrorP;
  };

  /**
   * @brief Add the states of the ancestor to the outgoing track
   * @param outTr The v1::Track that is being populated with information
   * @param secondAncestor The second ancestor, whose states are added to the outTr
   */
  void addStatesFromAncestor( LHCb::Event::v1::Track& outTr, const LHCb::Event::v1::Track& secondAncestor ) {
    for ( auto& state : secondAncestor.states() ) { outTr.addToStates( *state ); }
  }

  /**
   * @brief Create a track with two ancestors
   * @param inTrack The track to be converted
   * @param outTr The v1::Track that is being populated with information
   * @param ancestors1 The first set of ancestors
   * @param ancestors2 The second set of ancestors
   * @return A v1::Track
   */
  template <typename PrTrackType, typename ConversionInfo>
  LHCb::Event::v1::Track* createFromAncestor( const PrTrackType& inTrack, const LHCb::Event::v1::Tracks& ancestors1,
                                              const LHCb::Event::v1::Tracks& ancestors2 ) {
    auto ancTrack1 = ancestors1.object( inTrack.template get<typename ConversionInfo::Ancestor1>().cast() );
    auto ancTrack2 = ancestors2.object( inTrack.template get<typename ConversionInfo::Ancestor2>().cast() );
    auto outTrack  = new LHCb::Event::v1::Track( *ancTrack1 );
    outTrack->addToAncestors( ancTrack1 );
    outTrack->addToAncestors( ancTrack2 );
    if constexpr ( ConversionInfo::AddStatesFromAncestor ) addStatesFromAncestor( *outTrack, *ancTrack2 );
    return outTrack;
  }

  /**
   * @brief Create a track with one ancestor
   * @param inTrack The track to be converted
   * @param outTr The v1::Track that is being populated with information
   * @param ancestors1 The first set of ancestors
   * @return A v1::Track
   */
  template <typename PrTrackType, typename ConversionInfo>
  LHCb::Event::v1::Track* createFromAncestor( const PrTrackType& inTrack, const LHCb::Event::v1::Tracks& ancestors1 ) {
    auto ancTrack1 = ancestors1.object( inTrack.template get<typename ConversionInfo::Ancestor1>().cast() );
    auto outTrack  = new LHCb::Event::v1::Track( *ancTrack1 );
    outTrack->addToAncestors( ancTrack1 );
    return outTrack;
  }

  /**
   * @brief Create a track without ancestors
   * @param outTr The v1::Track that is being populated with information
   * @return A v1::Track
   */
  template <typename PrTrackType, typename ConversionInfo>
  LHCb::Event::v1::Track* createFromAncestor( const PrTrackType& /*inTrack*/ ) {
    auto outTrack = new LHCb::Event::v1::Track();
    return outTrack;
  }

  /**
   * @brief Helper function to assign input and output locations
   */

  template <typename Base, typename ConversionInfo, size_t... I>
  auto NamesHelper( std::index_sequence<I...> ) {
    using KV = typename Base::KeyValue;
    return std::make_tuple( KV{ "InputTracksLocation", "" }, KV{ ConversionInfo::AncestorLocations[I], "" }... );
  }
  template <typename Base, typename ConversionInfo, size_t... I>
  auto NamesHelper2( std::index_sequence<I...> ) {
    using KV = typename Base::KeyValue;
    return std::make_tuple( KV{ "InputTracksLocation1", "" }, KV{ "InputTracksLocation2", "" },
                            KV{ ConversionInfo::AncestorLocations[I], "" }... );
  }
  template <typename Base, typename ConversionInfo>
  struct IOHelper {
    static constexpr std::size_t NumInputs = ConversionInfo::AncestorLocations.size();
    static auto InputLocations() { return NamesHelper<Base, ConversionInfo>( std::make_index_sequence<NumInputs>{} ); }
    static auto OutputLocation() { return typename Base::KeyValue{ "OutputTracksLocation", "" }; }
  };

  /**
   * @brief Helper function to assign input and output locations for the merger
   */
  template <typename Base, typename ConversionInfo>
  struct IOHelperMerger {
    static constexpr std::size_t NumInputs = ConversionInfo::AncestorLocations.size();
    static auto InputLocations() { return NamesHelper2<Base, ConversionInfo>( std::make_index_sequence<NumInputs>{} ); }
    static auto OutputLocation() { return typename Base::KeyValue{ "OutputTracksLocation", "" }; }
  };

  /**
   * @brief Assign a momentum value to Velo tracks, given a fixed pT
   * @param outTr The v1::Track that is being populated with information
   * @param inTrack The track to be converted
   * @param veloPT The pT to be assigned to the velo tracks
   */
  template <typename PrTrackProxy>
  void calculateQOverPForVelo( const PrTrackProxy& inTrack, LHCb::Event::v1::Track* outTr, const float pTVelo ) {
    const int firstRow = outTr->lhcbIDs()[0].channelID();
    const int charge   = ( firstRow % 2 == 0 ? -1 : 1 );
    for ( auto& aState : outTr->states() ) {
      // -- Calculate the momentum per state
      const float tx1    = aState->tx();
      const float ty1    = aState->ty();
      const float slope2 = std::max( tx1 * tx1 + ty1 * ty1, 1.e-20f );
      const float qop    = charge * std::sqrt( slope2 ) / ( pTVelo * std::sqrt( 1.f + slope2 ) );
      aState->setQOverP( qop );
      aState->setErrQOverP2( 1e-6 );
      // -- fille the covariance
      auto                  covX = inTrack.StateCovX( aState->location() );
      auto                  covY = inTrack.StateCovY( aState->location() );
      Gaudi::TrackSymMatrix c;
      c( 0, 0 ) = covX.x().cast();
      c( 2, 0 ) = covX.y().cast();
      c( 2, 2 ) = covX.z().cast();
      c( 1, 1 ) = covY.x().cast();
      c( 3, 1 ) = covY.y().cast();
      c( 3, 3 ) = covY.z().cast();
      c( 4, 4 ) = 1.f;
      aState->setCovariance( c );
    }
  }

  /**
   * @brief Update the state of a track
   * @param outTr The v1::Track that is being populated with information
   * @param inTrack The track to be converted
   * @param stateLocation The location of the state to be updated
   */
  template <typename PrTrackProxy>
  void updateState( const PrTrackProxy& inTrack, LHCb::Event::v1::Track* outTr,
                    const LHCb::State::Location stateLocation ) {
    // update closest to beam state
    auto const qop = inTrack.qOverP();
    auto const pos = inTrack.closestToBeamStatePos();
    auto const dir = inTrack.closestToBeamStateDir();

    auto beamState = outTr->stateAt( stateLocation );
    beamState->setState( pos.x().cast(), pos.y().cast(), pos.z().cast(), dir.x().cast(), dir.y().cast(), qop.cast() );

    // update cov
    auto const covX = inTrack.covX();
    auto const covY = inTrack.covY();

    beamState->covariance()( 0, 0 ) = covX.x().cast();
    beamState->covariance()( 0, 2 ) = covX.y().cast();
    beamState->covariance()( 2, 2 ) = covX.z().cast();
    beamState->covariance()( 1, 1 ) = covY.x().cast();
    beamState->covariance()( 1, 3 ) = covY.y().cast();
    beamState->covariance()( 3, 3 ) = covY.z().cast();

    outTr->setChi2AndDoF( inTrack.chi2().cast(), inTrack.nDoF().cast() );
  }

  // ===========================================================================
  // -- The main converting function
  // ===========================================================================
  /**
   * @brief The actual converting function
   * @param outTr The v1::Track that is being populated with information
   * @param inTrack The track to be converted
   * @param errors The errors to be assigned to the outTr
   * @param veloPT The pT to be assigned to the velo tracks
   */
  template <typename PrTrackProxy, typename PrTracksType, typename ConversionInfo>
  void ConvertTrack( LHCb::Event::v1::Track* outTr, const PrTrackProxy& inTrack, const StateErrors& errors,
                     const float veloPT, const ITrackAddClusters* clusterAdder ) {

    constexpr auto stateLocations = ConversionInfo::StateLocations;
    constexpr bool isVeloTrack    = std::is_same_v<PrTracksType, LHCb::Pr::Velo::Tracks>;

    float QOverP    = 1.0f;
    float errQOverP = 1e-6f;

    if constexpr ( !isVeloTrack ) {
      QOverP    = inTrack.qOverP().cast();
      errQOverP = errors.stateErrorP * QOverP;
      // Adjust q/p and its uncertainty
      for ( auto& state : outTr->states() ) {
        state->covariance()( 4, 4 ) = errQOverP * errQOverP;
        state->setQOverP( QOverP );
      }
    }
    // Add LHCbIds
    // this gives _all_ hits on the track (also the ones in the ancestor),
    // but addToLhcbIDs checks for duplicates (assuming it is sorted)
    // as long as all tracks are converted with this code, sorting is guaranteed
    for ( auto const id : inTrack.lhcbIDs() ) { outTr->addToLhcbIDs( id ); }

    if ( clusterAdder ) {
      // Continuing execution even if some clusters are missing.
      clusterAdder->fillClustersFromLHCbIDs( *outTr ).ignore();
    }

    // -- some defaults
    outTr->setLikelihood( 999.9f );
    outTr->setGhostProbability( 999.9f );
    outTr->setPatRecStatus( LHCb::Event::v1::Track::PatRecStatus::PatRecIDs );

    for ( const auto loc : stateLocations ) {
      auto                  s = inTrack.StatePosDir( loc );
      Gaudi::TrackSymMatrix cov;
      cov( 0, 0 ) = errors.stateErrorX2;
      cov( 1, 1 ) = errors.stateErrorY2;
      cov( 2, 2 ) = errors.stateErrorTX2;
      cov( 3, 3 ) = errors.stateErrorTY2;
      cov( 4, 4 ) = errQOverP * errQOverP;

      outTr->addToStates(
          LHCb::State{ { s.x().cast(), s.y().cast(), s.tx().cast(), s.ty().cast(), QOverP }, cov, s.z().cast(), loc } );
    }
    // -- Velo tracks get an artifical momentum assigned, based on a constant pT
    if constexpr ( std::is_same_v<PrTracksType, LHCb::Pr::Velo::Tracks> && stateLocations.size() > 0 ) {
      calculateQOverPForVelo<PrTrackProxy>( inTrack, outTr, veloPT );
    }
  }
} // namespace

namespace LHCb::Converters::Track::v1 {

  template <typename PrTracksType, typename ConversionInfo, typename... VOneTracks>
  struct fromPrTracksV1Track
      : public Algorithm::Transformer<LHCb::Event::v1::Tracks( const PrTracksType&, const VOneTracks&... )> {

    using base_class_t = Algorithm::Transformer<LHCb::Event::v1::Tracks( const PrTracksType&, const VOneTracks&... )>;
    using V1Tracks     = LHCb::Event::v1::Tracks;

    fromPrTracksV1Track( std::string const& name, ISvcLocator* pSvcLocator )
        : base_class_t( name, pSvcLocator, IOHelper<base_class_t, ConversionInfo>::InputLocations(),
                        IOHelper<base_class_t, ConversionInfo>::OutputLocation() ) {}
    /**
     * @brief The main function
     * @param inTracks The set of input tracks
     * @param anecestorTracks The sets of tracks that serve as ancestors for inTracks
     * @return Keyed container with v1::Tracks
     */
    LHCb::Event::v1::Tracks operator()( const PrTracksType& inTracks,
                                        const VOneTracks&... ancestorTracks ) const override {

      auto outTracks = LHCb::Event::v1::Tracks{};
      outTracks.reserve( inTracks.size() );
      m_nbTracksCounter += inTracks.size();

      // -- It's not strictly speaking wrong, but it will add an additional uninitialized state, so let's issue a
      // warning.
      if constexpr ( std::is_same_v<ConversionInfo, LHCb::Pr::ConversionInfo::Velo> ) {
        if ( inTracks.backward() ) ++m_nbWrongVeloBackwardsConversion;
      }

      StateErrors errors{ m_stateErrorX2, m_stateErrorY2, m_stateErrorTX2, m_stateErrorTY2, m_stateErrorP };

      for ( auto const& inTrack : inTracks.scalar() ) {

        using TrackType = decltype( *std::declval<const PrTracksType>().scalar().begin() );
        auto outTrack   = createFromAncestor<TrackType, ConversionInfo>( inTrack, ancestorTracks... );
        //  ---
        ConvertTrack<TrackType, PrTracksType, ConversionInfo>( outTrack, inTrack, errors, m_veloPT,
                                                               m_clusterAdder.get() );
        if constexpr ( std::is_same_v<ConversionInfo, LHCb::Pr::ConversionInfo::Velo> )
          outTrack->setType( ConversionInfo::Type( inTrack.backward() ) );
        else
          outTrack->setType( ConversionInfo::Type );
        outTrack->setHistory( ConversionInfo::PrHistory );
        
        if constexpr( std::is_same_v<PrTracksType, LHCb::Pr::Seeding::Tracks> ){
          outTrack->setChi2PerDoF( {inTrack.chi2PerDoF().cast()} );
          outTrack->setNDoF({inTrack.nHits().cast()});
        }
        
        outTracks.insert( outTrack, inTrack.indices().cast() );
      }
      return outTracks;
    }

  private:
    ToolHandle<ITrackAddClusters> m_clusterAdder{ this, "TrackAddClusterTool", "AddClustersToTrackTool" };
    // - StateErrorX2: Error^2 on x-position (for making Track)
    Gaudi::Property<double> m_stateErrorX2{ this, "StateErrorX2", 4.0 };
    // - StateErrorY2: Error^2 on y-position (for making Track)
    Gaudi::Property<double> m_stateErrorY2{ this, "StateErrorY2", 400. };
    // - StateErrorTX2: Error^2 on tx-slope (for making Track)
    Gaudi::Property<double> m_stateErrorTX2{ this, "StateErrorTX2", 6.e-5 };
    // - StateErrorTY2: Error^2 on ty-slope (for making Track)
    Gaudi::Property<double> m_stateErrorTY2{ this, "StateErrorTY2", 1.e-4 };
    // - StateErrorP:  Error^2 on momentum (for making Track)
    Gaudi::Property<double> m_stateErrorP{ this, "StateErrorP", 0.15 };
    // - VeloPT:  Default PT for VeloTracks
    Gaudi::Property<double> m_veloPT{ this, "VeloPT", 400 * Gaudi::Units::MeV };
    // - A counter for the tracks
    mutable Gaudi::Accumulators::SummingCounter<>         m_nbTracksCounter{ this, "Nb of converted Tracks" };
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_nbWrongVeloBackwardsConversion{
        this, "Using Velo forward tracks conversion options on Velo backward tracks. Use at your own risk." };
  };
  using PrDownstreamConverter =
      fromPrTracksV1Track<LHCb::Pr::Downstream::Tracks, LHCb::Pr::ConversionInfo::Downstream, LHCb::Event::v1::Tracks>;
  DECLARE_COMPONENT_WITH_ID( PrDownstreamConverter, "fromPrDownstreamTracksV1Tracks" )
  using PrMatchConverter = fromPrTracksV1Track<LHCb::Pr::Long::Tracks, LHCb::Pr::ConversionInfo::Match,
                                               LHCb::Event::v1::Tracks, LHCb::Event::v1::Tracks>;
  DECLARE_COMPONENT_WITH_ID( PrMatchConverter, "fromPrMatchTracksV1Tracks" )
  using PrUpstreamConverter =
      fromPrTracksV1Track<LHCb::Pr::Upstream::Tracks, LHCb::Pr::ConversionInfo::Upstream, LHCb::Event::v1::Tracks>;
  DECLARE_COMPONENT_WITH_ID( PrUpstreamConverter, "fromPrUpstreamTracksV1Tracks" )
  using PrForwardConverter =
      fromPrTracksV1Track<LHCb::Pr::Long::Tracks, LHCb::Pr::ConversionInfo::Forward, LHCb::Event::v1::Tracks>;
  DECLARE_COMPONENT_WITH_ID( PrForwardConverter, "fromPrForwardTracksV1Tracks" )
  using PrForwardFromVeloUTConverter =
      fromPrTracksV1Track<LHCb::Pr::Long::Tracks, LHCb::Pr::ConversionInfo::ForwardFromVeloUT, LHCb::Event::v1::Tracks>;
  DECLARE_COMPONENT_WITH_ID( PrForwardFromVeloUTConverter, "fromPrForwardTracksFromVeloUTV1Tracks" )
  using PrSeedingConverter = fromPrTracksV1Track<LHCb::Pr::Seeding::Tracks, LHCb::Pr::ConversionInfo::Seeding>;
  DECLARE_COMPONENT_WITH_ID( PrSeedingConverter, "fromPrSeedingTracksV1Tracks" )
  using PrVeloConverter = fromPrTracksV1Track<LHCb::Pr::Velo::Tracks, LHCb::Pr::ConversionInfo::Velo>;
  DECLARE_COMPONENT_WITH_ID( PrVeloConverter, "fromPrVeloTracksV1Tracks" )
  using PrVeloBackwardConverter = fromPrTracksV1Track<LHCb::Pr::Velo::Tracks, LHCb::Pr::ConversionInfo::VeloBackward>;
  DECLARE_COMPONENT_WITH_ID( PrVeloBackwardConverter, "fromPrVeloBackwardTracksV1Tracks" )

  // -- This is the version when you need to merge two input containers with PrTracks into one output container
  // -- Mostly for Velo forward tracks + Velo backward tracks.
  template <typename PrTracksType, typename ConversionInfo1, typename ConversionInfo2, typename... VOneTracks>
  struct fromPrTracksV1TrackMerger : public Algorithm::Transformer<LHCb::Event::v1::Tracks(
                                         const PrTracksType&, const PrTracksType&, const VOneTracks&... )> {

    using base_class_t = Algorithm::Transformer<LHCb::Event::v1::Tracks( const PrTracksType&, const PrTracksType&,
                                                                         const VOneTracks&... )>;
    using V1Tracks     = LHCb::Event::v1::Tracks;

    fromPrTracksV1TrackMerger( std::string const& name, ISvcLocator* pSvcLocator )
        : base_class_t( name, pSvcLocator, IOHelperMerger<base_class_t, ConversionInfo1>::InputLocations(),
                        IOHelperMerger<base_class_t, ConversionInfo1>::OutputLocation() ) {}

    /**
     * @brief The main function
     * @param inTracks1 First set of input tracks
     * @param inTracks2 Second set of input tracks
     * @param anecestorTracks The sets of tracks that serve as ancestors for inTracks1 and inTracks2
     * @return Keyed container with v1::Tracks
     */
    LHCb::Event::v1::Tracks operator()( const PrTracksType& inTracks1, const PrTracksType& inTracks2,
                                        const VOneTracks&... ancestorTracks ) const override {

      auto outTracks = LHCb::Event::v1::Tracks{};
      outTracks.reserve( inTracks1.size() + inTracks2.size() );
      m_nbTracksCounter += inTracks1.size() + inTracks2.size();

      StateErrors errors{ m_stateErrorX2, m_stateErrorY2, m_stateErrorTX2, m_stateErrorTY2, m_stateErrorP };

      for ( auto const& inTrack : inTracks1.scalar() ) {

        using TrackType = decltype( *std::declval<const PrTracksType>().scalar().begin() );
        auto outTrack   = createFromAncestor<TrackType, ConversionInfo1>( inTrack, ancestorTracks... );
        //  ---
        ConvertTrack<TrackType, PrTracksType, ConversionInfo1>( outTrack, inTrack, errors, m_veloPT,
                                                                m_clusterAdder.get() );
        outTrack->setType( ConversionInfo1::Type );
        outTrack->setHistory( ConversionInfo1::PrHistory );
        outTracks.insert( outTrack );
      }

      for ( auto const& inTrack : inTracks2.scalar() ) {

        using TrackType = decltype( *std::declval<const PrTracksType>().scalar().begin() );
        auto outTrack   = createFromAncestor<TrackType, ConversionInfo2>( inTrack, ancestorTracks... );
        //  ---
        ConvertTrack<TrackType, PrTracksType, ConversionInfo2>( outTrack, inTrack, errors, m_veloPT,
                                                                m_clusterAdder.get() );
        outTrack->setType( ConversionInfo2::Type );
        outTrack->setHistory( ConversionInfo2::PrHistory );
        outTracks.insert( outTrack );
      }

      return outTracks;
    }

  private:
    ToolHandle<ITrackAddClusters> m_clusterAdder{ this, "TrackAddClusterTool", "AddClustersToTrackTool" };
    // - StateErrorX2: Error^2 on x-position (for making Track)
    Gaudi::Property<double> m_stateErrorX2{ this, "StateErrorX2", 4.0 };
    // - StateErrorY2: Error^2 on y-position (for making Track)
    Gaudi::Property<double> m_stateErrorY2{ this, "StateErrorY2", 400. };
    // - StateErrorTX2: Error^2 on tx-slope (for making Track)
    Gaudi::Property<double> m_stateErrorTX2{ this, "StateErrorTX2", 6.e-5 };
    // - StateErrorTY2: Error^2 on ty-slope (for making Track)
    Gaudi::Property<double> m_stateErrorTY2{ this, "StateErrorTY2", 1.e-4 };
    // - StateErrorP:  Error^2 on momentum (for making Track)
    Gaudi::Property<double> m_stateErrorP{ this, "StateErrorP", 0.15 };
    // - VeloPT:  Default PT for VeloTracks
    Gaudi::Property<double> m_veloPT{ this, "VeloPT", 400 * Gaudi::Units::MeV };
    // - A counter for the tracks
    mutable Gaudi::Accumulators::SummingCounter<> m_nbTracksCounter{ this, "Nb of converted Tracks" };
  };

  using PrVeloMergerConverter = fromPrTracksV1TrackMerger<LHCb::Pr::Velo::Tracks, LHCb::Pr::ConversionInfo::VeloForward,
                                                          LHCb::Pr::ConversionInfo::VeloBackward>;
  DECLARE_COMPONENT_WITH_ID( PrVeloMergerConverter, "fromPrVeloTracksV1TracksMerger" )

} // namespace LHCb::Converters::Track::v1
