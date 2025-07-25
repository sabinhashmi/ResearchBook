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
#include <limits>
#include <memory>
#include <optional>

#include "Gaudi/Accumulators.h"
#include "GaudiAlg/ISequencerTimerTool.h"

#include <Magnet/DeMagnet.h>

#include "LHCbAlgs/Transformer.h"
#include "LHCbMath/MatVec.h"
#include "LHCbMath/bit_cast.h"

#include "Event/PrLongTracks.h"
#include "Event/PrSeedTracks.h"
#include "Event/PrVeloTracks.h"
#include "Event/SOACollection.h"
#include "Event/StateParameters.h"
#include "Event/Track_v3.h"

#include "PrKernel/IPrAddUTHitsTool.h"
#include "PrKernel/IPrDebugTrackingTool.h"
#include "PrTrackModel.h"
#include "TrackInterfaces/ITrackMomentumEstimate.h"
#include "weights/TMVA_MLP_MatchNN_PrMatchNN.h"

//-----------------------------------------------------------------------------
// Implementation file for class : PrMatchNN
//
// 2013-11-15 : Michel De Cian, migration to Upgrade
//
// 2007-02-07 : Olivier Callot
//-----------------------------------------------------------------------------

/** @class PrMatchNN PrMatchNN.h
 *  Match Velo and Seed tracks
 *
 *  @author Michel De Cian (migration to Upgrade)
 *  @date 2013-11-15
 *
 *  @author Olivier Callot
 *  @date   2007-02-07
 */

namespace LHCb::Pr::MatchNN {
  namespace {
    using simd   = SIMDWrapper::best::types;
    using scalar = SIMDWrapper::scalar::types;

    using SeedTracks = LHCb::Pr::Seeding::Tracks;
    using VeloTracks = LHCb::Pr::Velo::Tracks;

    using VeloTag = LHCb::Pr::Velo::Tag;
    using SeedTag = LHCb::Pr::Seeding::Tag;

    namespace Tag {

      struct Index : LHCb::Event::int_field {};
      struct mlpVal : LHCb::Event::float_field {};

      struct veloIndex : LHCb::Event::int_field {};
      struct seedIndex : LHCb::Event::int_field {};

      template <typename T>
      using seedMLPPairs_t = LHCb::Event::SOACollection<T, Index, mlpVal>;

      template <typename T>
      using matchCandidates_t = LHCb::Event::SOACollection<T, veloIndex, seedIndex>;
    } // namespace Tag

    struct seedMLPPairs : Tag::seedMLPPairs_t<seedMLPPairs> {
      using base_t = typename Tag::seedMLPPairs_t<seedMLPPairs>;
      using base_t::base_t;
    };

    struct matchCandidates : Tag::matchCandidates_t<matchCandidates> {
      using base_t = typename Tag::matchCandidates_t<matchCandidates>;
      using base_t::base_t;
    };
  } // namespace

  template <typename SciFiTracks>
  class PrMatchNN : public Algorithm::Transformer<Long::Tracks( const Velo::Tracks&, const SciFiTracks&,
                                                                const IPrAddUTHitsTool&, const DeMagnet& ),
                                                  Algorithm::Traits::usesConditions<DeMagnet>> {

    using base_t   = Algorithm::Transformer<Long::Tracks( const Velo::Tracks&, const SciFiTracks&,
                                                          const IPrAddUTHitsTool&, const DeMagnet& ),
                                          Algorithm::Traits::usesConditions<DeMagnet>>;
    using KeyValue = typename base_t::KeyValue;

  public:
    //=============================================================================
    // Standard constructor, initializes variables
    //=============================================================================
    PrMatchNN( const std::string& name, ISvcLocator* pSvcLocator )
        : base_t( name, pSvcLocator,
                  { KeyValue{ "VeloInput", "Rec/Track/Velo" }, KeyValue{ "SeedInput", "Rec/Track/Seed" },
                    KeyValue{ "AddUTHitsToolName", "PrAddUTHitsTool" },
                    KeyValue{ "Magnet", LHCb::Det::Magnet::det_path } },
                  KeyValue{ "MatchOutput", "Rec/Track/Match" } )
        , m_NN{ inputVars } {}

    StatusCode initialize() override {
      return base_t::initialize().andThen( [this] {
        if ( m_timerTool.isEnabled() ) m_timerIndex = m_timerTool->addTimer( this->name() );
      } );
    }

    //  main method
    Long::Tracks operator()( const Velo::Tracks&, const SciFiTracks&, const IPrAddUTHitsTool&,
                             const DeMagnet& ) const override;

  private:
    static constexpr const std::array<const std::string_view, 6> inputVars = { "chi2",  "teta2",  "distX",
                                                                               "distY", "dSlope", "dSlopeY" };
    // calculate matching chi^2
    simd::mask_v checkChi2Match( const LinAlg::Vec<scalar::float_v, 3>        vState_pos,
                                 const LinAlg::Vec<scalar::float_v, 3>        vState_dir,
                                 const LinAlg::Vec<simd::float_v, 3>          sState_pos,
                                 const LinAlg::Vec<simd::float_v, 3>          sState_dir,
                                 std::array<simd::float_v, inputVars.size()>& mLPReaderInput,
                                 const VeloSciFiMatch<simd::float_v>& ) const;

    // merge velo and seed segment to output track
    Long::Tracks makeTracks( const Velo::Tracks& velos, const SciFiTracks& seeds, matchCandidates& matches,
                             const DeMagnet& magnet ) const;

    Gaudi::Property<float> m_zMatchY{ this, "zMatchY", 10000. * Gaudi::Units::mm };
    // -- Tolerances
    Gaudi::Property<float> m_dxTol{ this, "dxTol", 8. * Gaudi::Units::mm };
    Gaudi::Property<float> m_dxTolSlope{ this, "dxTolSlope", 80. * Gaudi::Units::mm };
    Gaudi::Property<float> m_dyTol{ this, "dyTol", 6. * Gaudi::Units::mm };
    Gaudi::Property<float> m_dyTolSlope{ this, "dyTolSlope", 300. * Gaudi::Units::mm };
    Gaudi::Property<float> m_fastYTol{ this, "FastYTol", 250. * Gaudi::Units::mm };
    // -- The main cut values
    Gaudi::Property<float> m_maxChi2{ this, "MaxMatchChi2", 15.0 };
    Gaudi::Property<float> m_minNN{ this, "MinMatchNN", 0.215 };
    Gaudi::Property<float> m_maxdDist{ this, "MaxdDist", 0.1 };
    Gaudi::Property<float> m_maxDistX{ this, "MaxDistX", 250 * Gaudi::Units::mm };
    Gaudi::Property<float> m_maxDistY{ this, "MaxDistY", 250 * Gaudi::Units::mm };
    Gaudi::Property<float> m_maxDSlope{ this, "MaxDSlope", 1.5 };
    Gaudi::Property<float> m_maxDSlopeY{ this, "MaxDSlopeY", 0.15 };
    Gaudi::Property<float> m_minP{ this, "MinP", 0.0 * Gaudi::Units::MeV };
    Gaudi::Property<float> m_minPt{ this, "MinPt", 0.0 * Gaudi::Units::MeV };

    // -- Counters
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_momentum_failed{ this, "momentum determination failed!" };
    mutable Gaudi::Accumulators::SummingCounter<unsigned int> m_tracksCount{ this, "#MatchingTracks" };
    mutable Gaudi::Accumulators::SummingCounter<float>        m_tracksMLP{ this, "#MatchingMLP" };
    mutable Gaudi::Accumulators::SummingCounter<float>        m_tracksChi2{ this, "#MatchingChi2" };

    ToolHandle<IPrDebugTrackingTool>   m_matchDebugTool{ this, "MatchDebugToolName", "" };
    ToolHandle<ITrackMomentumEstimate> m_fastMomentumTool{ this, "FastMomentumToolName", "FastMomentumEstimate" };
    mutable PublicToolHandle<ISequencerTimerTool> m_timerTool{ this, "TimerTool", "",
                                                               "Do not use in combination with multi-threading." };

    ReadMLPMatching m_NN;

    const simd::float_v m_dxTol2      = m_dxTol * m_dxTol;
    const simd::float_v m_dxTolSlope2 = m_dxTolSlope * m_dxTolSlope;
    const simd::float_v m_dyTol2      = m_dyTol * m_dyTol;
    const simd::float_v m_dyTolSlope2 = m_dyTolSlope * m_dyTolSlope;
    int                 m_timerIndex{};
  };

  // Declaration of the Algorithm Factory
  DECLARE_COMPONENT_WITH_ID( PrMatchNN<SeedTracks>, "PrMatchNN" )
  DECLARE_COMPONENT_WITH_ID( PrMatchNN<Event::v3::Tracks>, "PrMatchNNv3" )

  //=============================================================================
  // Main execution
  //=============================================================================
  template <typename SciFiTracks>
  Long::Tracks PrMatchNN<SciFiTracks>::operator()( const Velo::Tracks& velos, const SciFiTracks& seeds,
                                                   const IPrAddUTHitsTool& addUTHitsTool,
                                                   const DeMagnet&         magnet ) const {
    const auto scopedTimer = m_timerTool.get()->scopedTimer( m_timerIndex, m_timerTool.isEnabled() );
    std::array<simd::float_v, inputVars.size()> mLPReaderInput = {};

    if ( velos.size() == 0 || seeds.size() == 0 ) {
      return { nullptr, nullptr, nullptr, LHCb::Event::Enum::Track::History::PrMatch };
    }

    matchCandidates matches;
    seedMLPPairs    seedMLP;

    seedMLP.reserve( 3 );            // we rarely get more than 2 candidates per velo track
    matches.reserve( velos.size() ); // A bit more than half the velo tracks at most  make into long tracks

    // total chi2 for the event and the counters
    simd::float_v chi2Sum        = 0;
    auto          mlpCounterBuf  = m_tracksMLP.buffer();
    auto          chi2CounterBuf = m_tracksChi2.buffer();

    for ( auto const& velo : velos.scalar() ) {

      const auto velo_pos = velo.StatePos( Event::Enum::State::Location::EndVelo );
      const auto velo_dir = velo.StateDir( Event::Enum::State::Location::EndVelo );
      // sometimes consecutive clones come from the velo tracking, detect and reject these here
      if ( const auto offset = velo.offset(); offset > 0 ) {
        const auto prev_velo     = velos.scalar()[offset - 1];
        const auto prev_velo_pos = prev_velo.StatePos( Event::Enum::State::Location::EndVelo );
        const auto prev_velo_dir = prev_velo.StateDir( Event::Enum::State::Location::EndVelo );
        if ( essentiallyEqual( velo_pos.x().cast(), prev_velo_pos.x().cast() ) &&
             essentiallyEqual( velo_pos.y().cast(), prev_velo_pos.y().cast() ) &&
             essentiallyEqual( velo_dir.x().cast(), prev_velo_dir.x().cast() ) &&
             essentiallyEqual( velo_dir.y().cast(), prev_velo_dir.y().cast() ) ) {
          continue;
        }
      }
      const float posYApproxV    = velo_pos.y().cast() + ( m_zMatchY - velo_pos.z().cast() ) * velo_dir.y().cast();
      const auto  veloSciFiMatch = VeloSciFiMatch<simd::float_v>{
          velo_dir.x(), velo_dir.y(), velo_dir.x() * velo_dir.x(), velo_dir.y() * velo_dir.y() };

      seedMLP.clear();
      for ( auto const& s : seeds.simd() ) {

        const auto loopMask   = s.loop_mask();
        const auto seedidx    = s.indices();
        const auto seed_state = s.StatePosDir( Event::Enum::State::Location::EndT );
        const auto seed_pos   = LinAlg::Vec<simd::float_v, 3>{ seed_state.x(), seed_state.y(), seed_state.z() };
        // to avoid problems in the dSlope calculation later on, set out-of-range x slopes to 0.f
        const auto seed_dir =
            LinAlg::Vec<simd::float_v, 3>{ select( loopMask, seed_state.tx(), 0.f ), seed_state.ty(), 1.f };

        const auto posYApproxS = seed_state.y() + ( m_zMatchY.value() - seed_state.z() ) * seed_state.ty();
        if ( all( posYApproxS > posYApproxV + m_fastYTol.value() || !loopMask ) ) continue;

        const auto chi2Mask = checkChi2Match( velo_pos, velo_dir, seed_pos, seed_dir, mLPReaderInput, veloSciFiMatch );

        if ( none( chi2Mask && loopMask ) ) continue;

        chi2Sum += select( chi2Mask && loopMask, mLPReaderInput[0], 0 );
        const auto mlp = m_NN.GetMvaValue( mLPReaderInput );
        // Following is needed for mva training or data/mc comparison
        // For mc: option file to run is Moore/Hlt/RecoConf/options/tracking_developments/run_prmatching_debug.py
        if ( m_matchDebugTool.isEnabled() ) {
          const auto    state_beam = velo.template get<VeloTag::States>( 0 );
          simd::float_v qOverP, sigmaQOverP;
          m_fastMomentumTool
              ->calculate( magnet, seed_state.tx(), state_beam.tx(), state_beam.ty(), qOverP, sigmaQOverP, true )
              .orElse( [&] {
                ++m_momentum_failed;
                // assume the Velo/T station standalone reco do something reasonable
                qOverP = decltype( qOverP ){ std::numeric_limits<float>::quiet_NaN() };
              } )
              .ignore();
          auto selected =
              SIMDWrapper::to_array( select( chi2Mask && loopMask, mlp, std::numeric_limits<float>::quiet_NaN() ) );

          const auto velo_indices = SIMDWrapper::to_array<simd::int_v>( velo.offset() );
          const auto seed_indices = SIMDWrapper::to_array( seedidx );
          for ( size_t idx{ 0 }; idx < simd::size; ++idx ) {
            if ( std::isnan( selected[idx] ) ) continue;

            // prepare pairs of variable name and value
            std::array<IPrDebugTrackingTool::VariableDef, inputVars.size() + 9> vars_and_values;

            // for each variable name in inputVars get the value from the mlp input
            for ( auto [i, var] : LHCb::range::enumerate( inputVars ) ) {
              vars_and_values[i] = { var, SIMDWrapper::to_array( mLPReaderInput[i] )[idx] };
            }
            vars_and_values[inputVars.size()]     = { "quality",
                                                      m_matchDebugTool->check( velo_indices[idx], seed_indices[idx] ) };
            vars_and_values[inputVars.size() + 1] = { "mlp", SIMDWrapper::to_array( mlp )[idx] };
            vars_and_values[inputVars.size() + 2] = { "qop", SIMDWrapper::to_array( qOverP )[idx] };
            vars_and_values[inputVars.size() + 3] = { "redChi2", SIMDWrapper::to_array( s.chi2PerDoF() )[idx] };
            vars_and_values[inputVars.size() + 4] = { "tx", velo_dir.x().cast() };
            vars_and_values[inputVars.size() + 5] = { "ty", velo_dir.y().cast() };
            vars_and_values[inputVars.size() + 6] = { "tx_scifi", SIMDWrapper::to_array( seed_dir.x() )[idx] };
            vars_and_values[inputVars.size() + 7] = { "ty_scifi", SIMDWrapper::to_array( seed_dir.y() )[idx] };
            vars_and_values[inputVars.size() + 8] = { "qop_seed", SIMDWrapper::to_array( seed_state.qOverP() )[idx] };

            m_matchDebugTool->storeData( vars_and_values, "MVAInputAndOutput" );
          }
        }
        const auto mlpMask = mlp > m_minNN.value();
        auto       smlp = seedMLP.compress_back<SIMDWrapper::InstructionSet::Best>( mlpMask && chi2Mask && loopMask );
        smlp.template field<Tag::Index>().set( seedidx );
        smlp.template field<Tag::mlpVal>().set( mlp );

      } // end seed iter

      const auto best_proxy =
          std::max_element( seedMLP.scalar().begin(), seedMLP.scalar().end(), []( const auto& si, const auto& sj ) {
            return si.template get<Tag::mlpVal>() < sj.template get<Tag::mlpVal>();
          } );

      for ( auto const& s : seedMLP.scalar() ) {
        // keep only the ones that are close enough to best mlp
        if ( ( *best_proxy ).template get<Tag::mlpVal>().cast() - s.template get<Tag::mlpVal>().cast() <
             m_maxdDist.value() ) {
          auto match = matches.emplace_back<SIMDWrapper::InstructionSet::Scalar>();
          match.field<Tag::veloIndex>().set( velo.indices() );
          match.field<Tag::seedIndex>().set( s.template get<Tag::Index>().cast() );

          mlpCounterBuf += s.template get<Tag::mlpVal>().cast();
        }
      }

    } // end velo iter

    auto outputTracks = makeTracks( velos, seeds, matches, magnet );

    addUTHitsTool.addUTHits( outputTracks );

    m_tracksCount += outputTracks.size();
    chi2CounterBuf += chi2Sum.hadd();
    return outputTracks;
  }

  /**
   * @brief Calculates and checks a chi2-like value for the match of Velo and SciFi state.
   *
   * @param vState_pos Position 3-Vector of the Velo State.
   * @param vState_dir Slope 3-Vector of the Velo State.
   * @param sState_pos Position 3-Vector of the SciFi State.
   * @param sState_dir Slope 3-Vector of the SciFi State.
   * @param mLPReaderInput Array holding the values given to the NN.
   * @param vsMatch VeloSciFiMatch object holding parameterisations.
   * @return simd::mask_v Mask of passing checks.
   */
  template <typename SciFiTracks>
  simd::mask_v PrMatchNN<SciFiTracks>::checkChi2Match( const LinAlg::Vec<scalar::float_v, 3>        vState_pos,
                                                       const LinAlg::Vec<scalar::float_v, 3>        vState_dir,
                                                       const LinAlg::Vec<simd::float_v, 3>          sState_pos,
                                                       const LinAlg::Vec<simd::float_v, 3>          sState_dir,
                                                       std::array<simd::float_v, inputVars.size()>& mLPReaderInput,
                                                       const VeloSciFiMatch<simd::float_v>&         vsMatch ) const {

    const auto dSlopeAbs  = abs( sState_dir.x() - vState_dir.x() );
    const auto dSlopeYAbs = abs( sState_dir.y() - vState_dir.y() );
    const auto zMag       = vsMatch.calcZMagEndT( dSlopeAbs, sState_pos.x() );
    const auto xV         = vState_pos.x() + ( zMag - vState_pos.z() ) * vState_dir.x();
    const auto dSlope2    = dSlopeAbs * dSlopeAbs;
    const auto dSlopeY2   = dSlopeYAbs * dSlopeYAbs;
    const auto yV         = vState_pos.y() + ( m_zMatchY.value() - vState_pos.z() ) * vState_dir.y() +
                    vsMatch.calcYCorrMatch( dSlope2, dSlopeY2 );
    const auto xS    = sState_pos.x() + ( zMag - sState_pos.z() ) * sState_dir.x();
    const auto yS    = sState_pos.y() + ( m_zMatchY.value() - sState_pos.z() ) * sState_dir.y();
    const auto distX = abs( xS - xV );
    const auto distY = abs( yS - yV );
    const auto teta2 = vState_dir.x() * vState_dir.x() + vState_dir.y() * vState_dir.y();
    const auto tolX  = m_dxTol2 + dSlope2 * m_dxTolSlope2;
    const auto tolY  = m_dyTol2 + teta2 * m_dyTolSlope2;
    assert( all( abs( tolX ) > 0.f ) && all( abs( tolY ) > 0.f ) );
    // follwing is same as  chi2 += dSlopeY * dSlopeY / sState.errTy2() / 16.;
    // without division
    const auto chi2 = ( distX * distX / tolX ) + ( distY * distY / tolY ) + dSlopeY2 * 10000.f * 0.0625f;

    mLPReaderInput[0] = chi2;
    mLPReaderInput[1] = teta2;
    mLPReaderInput[2] = distX;
    mLPReaderInput[3] = distY;
    mLPReaderInput[4] = dSlopeAbs;
    mLPReaderInput[5] = dSlopeYAbs;

    return chi2 < m_maxChi2.value() && dSlopeAbs < m_maxDSlope.value() && dSlopeYAbs < m_maxDSlopeY.value() &&
           distX < m_maxDistX.value() && distY < m_maxDistY.value();
  }

  //=============================================================================
  template <typename SciFiTracks>
  Long::Tracks PrMatchNN<SciFiTracks>::makeTracks( const Velo::Tracks& velos, const SciFiTracks& seeds,
                                                   matchCandidates& matches, const DeMagnet& magnet ) const {
    auto result = [&] {
      if constexpr ( std::is_same_v<SciFiTracks, SeedTracks> ) {
        return Long::Tracks{ &velos, nullptr, &seeds, Event::Enum::Track::History::PrMatch };
      } else if constexpr ( std::is_same_v<SciFiTracks, Event::v3::Tracks> ) {
        // the v3 tracks do not work here as ancestors, this can be implemented if needed.
        return Long::Tracks{ &velos, nullptr, nullptr, Event::Enum::Track::History::PrMatch };
      }
    }();

    result.reserve( matches.size() );

    auto const seediter = seeds.simd();
    auto const veloiter = velos.simd();

    const bool          apply_min_p_cut = m_minP.value() > 0;
    const simd::float_v maxQOvP =
        ( apply_min_p_cut ? 1.0 / m_minP.value() : std::numeric_limits<simd::float_v>::max() );
    const bool apply_min_pt_cut = m_minPt.value() > 0;

    for ( auto const& match : matches.simd() ) {

      auto loopMask = match.loop_mask();

      auto const seed_track = seediter.gather( match.template get<Tag::seedIndex>(), loopMask );
      auto const velo_track = veloiter.gather( match.template get<Tag::veloIndex>(), loopMask );

      //== get Velo and T states at the usual pattern reco positions
      auto state_endvelo = velo_track.template get<VeloTag::States>( 1 );
      auto state_endT    = seed_track.StatePosDir( Event::Enum::State::Location::EndT );
      auto state_beam    = velo_track.template get<VeloTag::States>( 0 );

      //== estimate q/p
      simd::float_v qOverP, sigmaQOverP;
      m_fastMomentumTool
          ->calculate( magnet, state_endT.tx(), state_beam.tx(), state_beam.ty(), qOverP, sigmaQOverP, true )
          .orElse( [&] {
            ++m_momentum_failed;
            // assume the Velo/T station standalone reco do something reasonable
            qOverP = -std::numeric_limits<simd::float_v>::max(); // what is a good nonsense value
          } )
          .ignore();

      // If enabled filter on min track P, Pt
      if ( apply_min_p_cut ) {
        auto pMask = abs( qOverP ) < maxQOvP;
        loopMask   = loopMask && pMask;
      }
      if ( apply_min_pt_cut ) {
        const auto momentum = abs( 1.0 / qOverP );
        const auto tx2      = state_beam.tx() * state_beam.tx();
        const auto ty2      = state_beam.ty() * state_beam.ty();
        const auto sinTrack = sqrt( 1.0 - ( 1.0 / ( 1.0 + tx2 + ty2 ) ) );
        const auto pt       = sinTrack * momentum;
        auto       ptMask   = m_minPt.value() < pt;
        loopMask            = loopMask && ptMask;
      }

      auto const oTrack = result.template compress_back<SIMDWrapper::InstructionSet::Best>( loopMask );
      using TracksTag   = LHCb::Pr::Long::Tag;
      oTrack.template field<TracksTag::trackVP>().set( match.get<Tag::veloIndex>() );
      oTrack.template field<TracksTag::trackUT>().set( -1 );
      oTrack.template field<TracksTag::trackSeed>().set( match.get<Tag::seedIndex>() );

      auto const n_fthits = seed_track.nHits();
      auto const n_vphits = velo_track.nHits();
      oTrack.template field<TracksTag::VPHits>().resize( n_vphits );
      oTrack.template field<TracksTag::UTHits>().resize( 0 );
      oTrack.template field<TracksTag::FTHits>().resize( n_fthits );

      for ( auto idx{ 0 }; idx < n_vphits.hmax( loopMask ); ++idx ) {
        oTrack.template field<TracksTag::VPHits>()[idx].template field<TracksTag::Index>().set(
            velo_track.vp_index( idx ) );
        oTrack.template field<TracksTag::VPHits>()[idx].template field<TracksTag::LHCbID>().set(
            velo_track.vp_lhcbID( idx ) );
      }
      for ( auto idx{ 0 }; idx < n_fthits.hmax( loopMask ); ++idx ) {
        const auto ft_index = [&] {
          if constexpr ( std::is_same_v<SciFiTracks, SeedTracks> ) {
            return seed_track.ft_index( idx );
          } else if constexpr ( std::is_same_v<SciFiTracks, Event::v3::Tracks> ) {
            // the v3 tracks do not store the index, if this is needed it can be implemented
            return -1;
          }
        }();
        oTrack.template field<TracksTag::FTHits>()[idx].template field<TracksTag::Index>().set( ft_index );
        oTrack.template field<TracksTag::FTHits>()[idx].template field<TracksTag::LHCbID>().set(
            seed_track.ft_lhcbID( idx ) );
      }

      static_assert( Event::v3::num_states<Event::Enum::Track::Type::Long>() == 2 );
      constexpr auto endVelo = stateIndex<Event::Enum::Track::Type::Long>( Event::Enum::State::Location::EndVelo );
      constexpr auto endT    = stateIndex<Event::Enum::Track::Type::Long>( Event::Enum::State::Location::EndT );

      // store end of VELO state
      oTrack.template field<TracksTag::States>( endVelo ).setPosition( state_endvelo.x(), state_endvelo.y(),
                                                                       state_endvelo.z() );
      oTrack.template field<TracksTag::States>( endVelo ).setDirection( state_endvelo.tx(), state_endvelo.ty() );
      oTrack.template field<TracksTag::States>( endVelo ).setQOverP( qOverP );

      oTrack.template field<TracksTag::States>( endT ).setPosition( state_endT.x(), state_endT.y(), state_endT.z() );
      oTrack.template field<TracksTag::States>( endT ).setDirection( state_endT.tx(), state_endT.ty() );
      oTrack.template field<TracksTag::States>( endT ).setQOverP( qOverP );
    }

    return result;
  }
  //=============================================================================

} // namespace LHCb::Pr::MatchNN
