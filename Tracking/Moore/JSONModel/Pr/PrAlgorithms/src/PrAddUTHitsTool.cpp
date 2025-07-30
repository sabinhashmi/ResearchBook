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

#include "PrKernel/IPrAddUTHitsTool.h" // Interface
#include "PrKernel/PrMutUTHits.h"

#include "DetDesc/GenericConditionAccessorHolder.h"
#include "Event/PrHits.h"
#include "Event/PrLongTracks.h"
#include "Event/State.h"
#include "Event/UTSectorHelper.h"
#include "Event/UTTrackUtils.h"
#include "LHCbAlgs/Traits.h"
#include "Magnet/DeMagnet.h"

#include "Kernel/LHCbID.h"
#include "Kernel/STLExtensions.h"
#include "LHCbMath/GeomFun.h"
#include "LHCbMath/SIMDWrapper.h"
#include "UTDAQ/UTDAQHelper.h"
#include "UTDAQ/UTInfo.h"

#include "GaudiAlg/FunctionalTool.h"
#include "GaudiAlg/GaudiTool.h"
#include "GaudiKernel/DataObjectHandle.h"
#include "GaudiKernel/IBinder.h"
#include "GaudiKernel/SystemOfUnits.h"

#include "boost/container/small_vector.hpp"
#include "boost/container/static_vector.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <optional>

//-----------------------------------------------------------------------------
// Implementation file for class : PrAddUTHitsTool
//-----------------------------------------------------------------------------
/**
 * @class PrAddUTHitsTool PrAddUTHitsTool.h
 *
 * \brief  Adds UT hits to long tracks, see note LHCb-INT-2010-20 for TT version
 *
 * Parameters:
 * - ZUTField: Z-Position of the kink for the state extrapolation
 * - ZMSPoint: Z-Position of the multiple scattering point
 * - UTParam: Parameter of the slope of the state extrapolation
 * - MaxChi2Tol: Offset of the chi2 cut
 * - MaxChi2Slope: Slope of the chi2 cut
 * - MaxChi2POffset: Momentum offest of the chi2 cut
 * - YTolSlope: Offest of the y-tolerance cut
 * - XTol: Offest of the x-window cut
 * - XTolSlope: Slope of the x-window cut
 * - MajAxProj: Major axis of the ellipse for the cut on the projection
 * - MinAxProj: Minor axis of the ellipse for the cut on the projection
 * - ZUTProj: Z-Position which state has to be closest to
 *
 *
 *  @author Michel De Cian
 *  @date   2016-05-11
 *
 */
namespace {
  namespace HitTag    = LHCb::Pr::UT::Mut::HitTag;
  namespace UTHitsTag = LHCb::Pr::UT::UTHitsTag;
  using TracksTag     = LHCb::Pr::Long::Tag;
  using boost::container::small_vector;
  using boost::container::static_vector;
  using LHCb::UT::TrackUtils::BoundariesNominal;
  using LHCb::UT::TrackUtils::MiniStates;
  using ROOT::Math::CholeskyDecomp;
  using LongTracks = LHCb::Pr::Long::Tracks;

  // number of hits roughly matching a long track found in UT layers, on average in pp 9 +- 7
  constexpr auto nHitsInLayers = 64;
  constexpr auto MaxUTHits     = LHCb::Pr::Upstream::Tracks::MaxUTHits;
} // namespace

namespace LHCb::Pr {

  using simd   = SIMDWrapper::best::types;
  using I      = simd::int_v;
  using F      = simd::float_v;
  using scalar = SIMDWrapper::scalar::types;
  using sI     = scalar::int_v;
  using sF     = scalar::float_v;

  constexpr auto totUTLayers = static_cast<int>( UTInfo::DetectorNumbers::TotalLayers );
  constexpr auto UTLayers    = static_cast<int>( UTInfo::DetectorNumbers::Layers );

  class PrAddUTHitsTool
      : public Gaudi::Functional::ToolBinder<
            Gaudi::Interface::Bind::Box<IPrAddUTHitsTool>( const LHCb::UTDAQ::GeomCache&, const UT::Hits&,
                                                           const DeMagnet& ),
            LHCb::Algorithm::Traits::usesBaseAndConditions<FixTESPath<AlgTool>, LHCb::UTDAQ::GeomCache, DeMagnet>> {
  private:
    class BoundInstance final : public Gaudi::Interface::Bind::Stub<IPrAddUTHitsTool> {

      const PrAddUTHitsTool*        m_parent = nullptr;
      const LHCb::UTDAQ::GeomCache& m_geomcache;
      const UT::Hits&               m_HitHandler;
      const DetElementRef<DeMagnet> m_magnet;

    public:
      BoundInstance( const PrAddUTHitsTool* parent, const LHCb::UTDAQ::GeomCache& geomcache, const UT::Hits& hitHandler,
                     const DeMagnet& magnet )
          : m_parent{ parent }, m_geomcache{ geomcache }, m_HitHandler{ hitHandler }, m_magnet{ magnet } {}

      void addUTHits( LongTracks& tracks ) const override {
        m_parent->addUTHits( tracks, m_geomcache, m_HitHandler, *m_magnet );
      }

      void getUTHits( const LHCb::State& state, static_vector<LHCb::LHCbID, MaxUTHits>& hits ) const override {
        m_parent->getUTHits( state, hits, m_geomcache, m_HitHandler, *m_magnet );
      }
    };

  public:
    PrAddUTHitsTool( std::string type, std::string name, const IInterface* parent )
        : ToolBinder{ std::move( type ),
                      name,
                      parent,
                      { { "UTGeomCache", "AlgorithmSpecific-" + name + "UTGeomCache" },
                        { "UTHitsLocation", "UT/PrUTHits" },
                        { "Magnet", LHCb::Det::Magnet::det_path } },
                      construct<BoundInstance>( this )

          } {}

    StatusCode initialize() override {
      return ToolBinder::initialize().andThen( [&] {
        addConditionDerivation<LHCb::UTDAQ::GeomCache( const DeUTDetector& )>(
            { DeUTDetLocation::location() }, inputLocation<LHCb::UTDAQ::GeomCache>() );
      } );
    }

    /** @brief Add UT clusters to matched tracks. This calls findUTHits internally
        @param track Track to add the UT hits to
    */
    void addUTHits( LongTracks&, const LHCb::UTDAQ::GeomCache&, const UT::Hits&, const DeMagnet& ) const;
    /** Return UT hits without adding them.
        @param state State closest to UT for extrapolation (normally Velo state)
        @param ttHits Container to fill UT hits in
    */
    void getUTHits( const LHCb::State&, static_vector<LHCb::LHCbID, MaxUTHits>&, const LHCb::UTDAQ::GeomCache&,
                    const UT::Hits&, const DeMagnet& ) const;

  private:
    Gaudi::Property<unsigned> m_minUThits{ this, "MinUTHits", 2 };
    Gaudi::Property<float>    m_zUTField{ this, "ZUTField", 1740. * Gaudi::Units::mm };
    Gaudi::Property<float>    m_zMSPoint{ this, "ZMSPoint", 400. * Gaudi::Units::mm };
    Gaudi::Property<float>    m_utParam{ this, "UTParam", 29. };
    Gaudi::Property<float>    m_zUTProj{ this, "ZUTProj", 2500. * Gaudi::Units::mm };
    Gaudi::Property<float>    m_maxChi2Tol{ this, "MaxChi2Tol", 2.0 };
    Gaudi::Property<float>    m_maxChi2Slope{ this, "MaxChi2Slope", 25000.0 };
    Gaudi::Property<float>    m_maxChi2POffset{ this, "MaxChi2POffset", 100.0 };
    Gaudi::Property<float>    m_yTolSlope{ this, "YTolSlope", 20000. };
    Gaudi::Property<float>    m_xTol{ this, "XTol", 1.0 };
    Gaudi::Property<float>    m_xTolSlope{ this, "XTolSlope", 30000.0 };
    Gaudi::Property<float>    m_fixedWeightFactor{ this, "FixedWeightFactor", 9.f };
    float                     m_invMajAxProj2 = 0.0;
    Gaudi::Property<float>    m_majAxProj{
        this, "MajAxProj", 20.0 * Gaudi::Units::mm,
        [this]( auto& ) { this->m_invMajAxProj2 = 1 / ( this->m_majAxProj * this->m_majAxProj ); },
        Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };
    Gaudi::Property<float> m_minAxProj{ this, "MinAxProj", 2.0 * Gaudi::Units::mm };
    Gaudi::Property<bool>  m_enableTool{ this, "EnableTool", true };

    mutable Gaudi::Accumulators::SummingCounter<> m_hitsAddedCounter{ this, "#UT hits added" };
    mutable Gaudi::Accumulators::Counter<>        m_tracksWithHitsCounter{ this, "#tracks with hits added" };

    template <typename InputType>
    std::array<BoundariesNominal, totUTLayers> findSectors( const InputType&, MiniStates&, float,
                                                            const LHCb::UTDAQ::GeomCache& ) const;

    template <typename ProxyState>
    static_vector<int, MaxUTHits> findUTHits( const ProxyState&, const UT::Mut::Hits&, LHCb::span<const int> ) const;

    template <typename ProxyState>
    bool selectHits( const ProxyState&, const std::array<BoundariesNominal, totUTLayers>&, UT::Mut::Hits&,
                     const UT::Hits&, float ) const;
    template <typename Container>
    float calculateChi2( float, float, unsigned, Container&, const UT::Mut::Hits& ) const;
  };
  // Declaration of the Algorithm Factory
  DECLARE_COMPONENT_WITH_ID( PrAddUTHitsTool, "PrAddUTHitsTool" )

  //=========================================================================
  //  Add the UT hits on the track, only the ids.
  //=========================================================================
  void PrAddUTHitsTool::addUTHits( LongTracks& tracks, const LHCb::UTDAQ::GeomCache& geomcache,
                                   const UT::Hits& hitHandler, const DeMagnet& magnet ) const {

    if ( !m_enableTool ) return;

    auto hitsAddedCounter      = m_hitsAddedCounter.buffer();
    auto tracksWithHitsCounter = m_tracksWithHitsCounter.buffer();

    const auto signedReCur = magnet.signedRelativeCurrent();

    MiniStates filteredStates;
    filteredStates.reserve( tracks.size() );

    // avoid FPEs in findSectors in Brunel
    // TODO: nuke Brunel
    if ( const auto size = tracks.size(); size % simd::size ) {
      if ( ( size + simd::size ) > tracks.capacity() ) tracks.reserve( size + simd::size );
      tracks.simd()[size].field<TracksTag::States>( 0 ).setQOverP( 1.f );
      tracks.simd()[size].field<TracksTag::States>( 0 ).setPosition( LHCb::LinAlg::Vec<F, 3>{ 1.f, 1.f, 1.f } );
      tracks.simd()[size].field<TracksTag::States>( 0 ).setDirection( LHCb::LinAlg::Vec<F, 3>{ 1.f, 1.f, 1.f } );
    }

    const auto compBoundsArray = findSectors( tracks, filteredStates, signedReCur, geomcache );

    // -- Define the container for all the hits compatible with the track
    UT::Mut::Hits hitsInLayers;
    hitsInLayers.reserve( nHitsInLayers );
    small_vector<int, nHitsInLayers> permutations{};

    for ( const auto& fState : filteredStates.scalar() ) {

      if ( !selectHits( fState, compBoundsArray, hitsInLayers, hitHandler, signedReCur ) ) continue;
      permutations.clear();
      for ( unsigned i{ 0 }; i < hitsInLayers.size(); ++i ) permutations.push_back( i );

      const auto& hitsInL = hitsInLayers.scalar();
      std::sort( permutations.begin(), permutations.end(), [&hitsInL]( auto i, auto j ) {
        return hitsInL[i].projection().cast() < hitsInL[j].projection().cast();
      } );

      // there should be no duplicates
      assert( std::adjacent_find( permutations.begin(), permutations.end(), [&hitsInL]( auto i, auto j ) {
                return hitsInL[i].channelID().cast() == hitsInL[j].channelID().cast();
              } ) == permutations.end() );

      const auto myUTHits = findUTHits( fState, hitsInLayers, permutations );

      if ( myUTHits.size() < m_minUThits ) continue;

      const auto itr     = fState.get<LHCb::UT::TrackUtils::MiniStateTag::index>().cast();
      const auto nUTHits = static_cast<sI>( myUTHits.size() );
      tracks.scalar()[itr].field<TracksTag::UTHits>().resize( nUTHits );
      for ( auto [offset, iHit] : LHCb::range::enumerate( myUTHits ) ) {
        const auto myUTHit = hitsInL[iHit];
        const auto utID    = static_cast<LHCb::Detector::UT::ChannelID>( myUTHit.channelID().cast() );
        tracks.scalar()[itr].field<TracksTag::UTHits>()[offset].template field<TracksTag::Index>().set(
            myUTHit.index() );
        tracks.scalar()[itr].field<TracksTag::UTHits>()[offset].template field<TracksTag::LHCbID>().set(
            LHCb::Event::lhcbid_v<SIMDWrapper::scalar::types>{ LHCb::LHCbID{ utID } } );
      }
      hitsAddedCounter += myUTHits.size();
      tracksWithHitsCounter++;
    }
  }
  ///=======================================================================
  //  find UT hits for a input state
  ///=======================================================================
  void PrAddUTHitsTool::getUTHits( const LHCb::State& veloState, static_vector<LHCb::LHCbID, MaxUTHits>& uthits,
                                   const LHCb::UTDAQ::GeomCache& geomcache, const UT::Hits& hitHandler,
                                   const DeMagnet& magnet ) const {
    if ( !m_enableTool ) return;
    const auto signedReCur = magnet.signedRelativeCurrent();

    MiniStates filteredStates;
    filteredStates.reserve( simd::size );

    UT::Mut::Hits hitsInLayers;
    hitsInLayers.reserve( nHitsInLayers );

    const auto compBoundsArray = findSectors( veloState, filteredStates, signedReCur, geomcache );
    if ( !filteredStates.size() ) return;

    const auto& fState = filteredStates.scalar()[0];
    selectHits( fState, compBoundsArray, hitsInLayers, hitHandler, signedReCur );

    small_vector<int, nHitsInLayers> permutations{};
    for ( unsigned i{ 0 }; i < hitsInLayers.size(); ++i ) permutations.push_back( i );
    const auto& hitsInL = hitsInLayers.scalar();
    std::sort( permutations.begin(), permutations.end(), [&hitsInL]( auto i, auto j ) {
      return hitsInL[i].projection().cast() < hitsInL[j].projection().cast();
    } );

    assert( std::adjacent_find( permutations.begin(), permutations.end(), [&hitsInL]( auto i, auto j ) {
              return hitsInL[i].channelID().cast() == hitsInL[j].channelID().cast();
            } ) == permutations.end() );

    const auto myUTHits = findUTHits( fState, hitsInLayers, permutations );

    for ( auto iHit : myUTHits ) {
      LHCb::LHCbID lhcbid( LHCb::Detector::UT::ChannelID( hitsInL[iHit].channelID().cast() ) );
      uthits.push_back( lhcbid );
    }
  }

  ///=======================================================================
  //  find all sections
  ///=======================================================================
  template <typename InputType>
  std::array<BoundariesNominal, totUTLayers>
  PrAddUTHitsTool::findSectors( const InputType& inputs, MiniStates& filteredStates, float signedReCur,
                                const LHCb::UTDAQ::GeomCache& geomcache ) const {

    if constexpr ( std::is_same_v<InputType, LongTracks> ) {
      for ( auto const& track : inputs.simd() ) {
        //---Define the tolerance parameters
        const auto qoverp = track.qOverP();
        const auto pos    = track.StatePos( Event::Enum::State::Location::EndVelo );
        const auto dir    = track.StateDir( Event::Enum::State::Location::EndVelo );

        auto fState = filteredStates.emplace_back<SIMDWrapper::InstructionSet::Best>();
        fState.field<LHCb::UT::TrackUtils::MiniStateTag::State>().setPosition( pos.x(), pos.y(), pos.z() );
        fState.field<LHCb::UT::TrackUtils::MiniStateTag::State>().setDirection( dir.x(), dir.y() );
        fState.field<LHCb::UT::TrackUtils::MiniStateTag::State>().setQOverP( qoverp );
        fState.field<LHCb::UT::TrackUtils::MiniStateTag::index>().set( track.indices() );
      }
    } else {
      auto fState = filteredStates.emplace_back<SIMDWrapper::InstructionSet::Scalar>();
      fState.field<LHCb::UT::TrackUtils::MiniStateTag::State>().setPosition( inputs.x(), inputs.y(), inputs.z() );
      fState.field<LHCb::UT::TrackUtils::MiniStateTag::State>().setDirection( inputs.tx(), inputs.ty() );
      fState.field<LHCb::UT::TrackUtils::MiniStateTag::State>().setQOverP( inputs.qOverP() );
      fState.field<LHCb::UT::TrackUtils::MiniStateTag::index>().set( 0 );
    }

    auto compBoundsArray =
        LHCb::UTDAQ::findAllSectorsExtrap<BoundariesNominal, LHCb::UT::TrackUtils::BoundariesNominalTag::types,
                                          LHCb::UT::TrackUtils::maxNumRowsBoundariesNominal,
                                          LHCb::UT::TrackUtils::maxNumColsBoundariesNominal>(
            filteredStates, geomcache,
            [&]( int layerIndex, simd::float_v x, simd::float_v y, simd::float_v z, simd::float_v tx, simd::float_v ty,
                 simd::float_v qop ) {
              const auto bendParam = m_utParam * -signedReCur * qop;
              const auto overp     = abs( qop );

              const simd::float_v xTol = m_xTol + m_xTolSlope.value() * overp;
              const simd::float_v yTol = m_yTolSlope.value() * overp;

              const simd::float_v zLayer{ geomcache.layers[layerIndex].z };
              const simd::float_v dxDy{ geomcache.layers[layerIndex].dxDy };

              const simd::float_v xLayer = x + ( zLayer - z ) * tx + bendParam * ( zLayer - m_zUTField.value() );
              const simd::float_v yLayer = y + ( zLayer - z ) * ty;
              return std::make_tuple( xLayer, yLayer, xTol, yTol );
            } );

    return compBoundsArray;
  }

  //=========================================================================
  //  Return the UT hits
  //=========================================================================
  template <typename ProxyState>
  static_vector<int, MaxUTHits> PrAddUTHitsTool::findUTHits( const ProxyState&     fState,
                                                             const UT::Mut::Hits&  hitsInLayers,
                                                             LHCb::span<const int> permutations ) const {

    const auto p = std::abs( 1.f / fState.template get<LHCb::UT::TrackUtils::MiniStateTag::State>().qOverP().cast() );
    auto       bestChi2 = m_maxChi2Tol + m_maxChi2Slope / ( p - m_maxChi2POffset );

    const auto& hitsInL = hitsInLayers.scalar();

    small_vector<int, 3 * MaxUTHits> goodUT{};
    static_vector<int, MaxUTHits>    UTHits{};
    // -- Loop over all hits and make "groups" of hits to form a candidate
    // -- allow 'groups' of hits with a minimal size according to minimum of UT hits
    auto itB_offsets = ( m_minUThits < 3 ) ? static_vector<unsigned, 2>{ 2u, 1u } : static_vector<unsigned, 2>{ 2u };
    for ( unsigned itB_offset : itB_offsets ) {
      // start with 3-hit combinations then try 2-hit combinations if that fails and is asked for
      if ( UTHits.size() > 2 ) break;
      for ( unsigned itB{ 0 }; itB + itB_offset < permutations.size(); ++itB ) {
        const auto itBeg     = permutations[itB];
        const auto firstProj = hitsInL[itBeg].projection().cast();
        goodUT.clear();
        // -- If |firstProj| > m_majAxProj, the sqrt is ill defined
        const auto maxProj =
            std::abs( firstProj ) < m_majAxProj
                ? firstProj + std::sqrt( m_minAxProj * m_minAxProj * ( 1.f - firstProj * firstProj * m_invMajAxProj2 ) )
                : firstProj;

        if ( hitsInL[permutations[itB + itB_offset]].projection().cast() > maxProj ) continue;

        auto nbPlane     = 0u;
        auto firedPlanes = std::array{ 0, 0, 0, 0 };
        // -- Make "group" of hits which are within a certain distance to the first hit of the group
        for ( auto itE = itB; itE < permutations.size(); itE++ ) {
          const auto itEnd = permutations[itE];
          if ( hitsInL[itEnd].projection().cast() > maxProj ) break;
          if ( const auto pc = hitsInL[itEnd].planeCode().cast(); !firedPlanes[pc] ) {
            ++firedPlanes[pc]; // -- Count number of fired planes
            ++nbPlane;
          }
          goodUT.push_back( itEnd );
        }
        // -- group of hits has to be at least as large than best group at this stage
        if ( nbPlane < m_minUThits || UTHits.size() > goodUT.size() )
          continue; // -- Need at least hits in 'm_minUThits' planes

        // ----------------------------------
        if ( msgLevel( MSG::DEBUG ) ) {
          debug() << "Start fit, first proj " << firstProj << " nbPlane " << nbPlane << " size " << goodUT.size()
                  << endmsg;
        }

        const auto chi2 = calculateChi2( bestChi2, p, UTHits.size(), goodUT, hitsInLayers );

        // -- If this group has a better (and well defined) chi2 than all the others
        // -- and is at least as large as all the others, then make this group the new candidate
        // -- prefer 3 hits over 2 (to partly recover min 3 hit case behaviour) by first running over all 3 hit cases
        if ( chi2 > 0. && bestChi2 > chi2 && goodUT.size() >= UTHits.size() ) {
          UTHits.clear();
          std::copy( goodUT.begin(), goodUT.end(), std::back_inserter( UTHits ) );
          bestChi2 = chi2;
        }
      }
    }
    return UTHits;
  }
  //=========================================================================
  // Select the hits in a certain window
  //=========================================================================
  template <typename ProxyState>
  bool PrAddUTHitsTool::selectHits( const ProxyState&                                 fState,
                                    const std::array<BoundariesNominal, totUTLayers>& compBoundsArray,
                                    UT::Mut::Hits& hitsInLayers, const UT::Hits& hitHandler, float signedReCur ) const {

    hitsInLayers.clear();
    // -- This is for some sanity checks later
    [[maybe_unused]] constexpr const auto maxSectorNumber = 2304;
    // With the new UT chID definition the sector number can go up to 2304 given the following equation
    //     ( ( ( ( ( ( side * static_cast<int>( UTInfo::DetectorNumbers::HalfLayers ) + halflayer ) *
    //                          static_cast<int>( UTInfo::DetectorNumbers::Staves ) ) +
    //                        stave ) *
    //                          static_cast<int>( UTInfo::DetectorNumbers::Faces ) +
    //                      face ) *
    //                    static_cast<int>( UTInfo::DetectorNumbers::Modules ) ) +
    //                  module ) *
    //                    static_cast<int>( UTInfo::DetectorNumbers::SubSectors ) +
    //                subsector;

    const auto stateX    = fState.template get<LHCb::UT::TrackUtils::MiniStateTag::State>().x().cast();
    const auto stateY    = fState.template get<LHCb::UT::TrackUtils::MiniStateTag::State>().y().cast();
    const auto stateZ    = fState.template get<LHCb::UT::TrackUtils::MiniStateTag::State>().z().cast();
    const auto stateTx   = fState.template get<LHCb::UT::TrackUtils::MiniStateTag::State>().tx().cast();
    const auto stateTy   = fState.template get<LHCb::UT::TrackUtils::MiniStateTag::State>().ty().cast();
    const auto qop       = fState.template get<LHCb::UT::TrackUtils::MiniStateTag::State>().qOverP().cast();
    const auto p         = std::abs( 1.f / qop );
    const auto bendParam = m_utParam * -signedReCur * qop;

    if ( msgLevel( MSG::DEBUG ) )
      debug() << "State z:  " << stateZ << " x " << stateX << " y " << stateY << " tx " << stateTx << " ty " << stateTy
              << " p " << p << endmsg;

    std::size_t nSize   = 0;
    std::size_t nLayers = 0;
    const auto& myHits  = hitHandler;
    for ( auto&& [layerIndex, compBoundsLayer] : LHCb::range::enumerate( compBoundsArray ) ) {
      if ( ( layerIndex == 2 && nLayers == 0 ) || ( layerIndex == 3 && nLayers < 2 ) ) return false;

      // -- Define the tolerance parameters
      const auto yTol = m_yTolSlope / p;
      const auto xTol = m_xTol + m_xTolSlope / p;

      const auto& boundsarr = compBoundsLayer.scalar();
      const auto  nPos =
          boundsarr[fState.offset()].template get<LHCb::UT::TrackUtils::BoundariesNominalTag::nPos>().cast();

      for ( auto j = 0; j < nPos; ++j ) {

        const auto sectA =
            boundsarr[fState.offset()].template get<LHCb::UT::TrackUtils::BoundariesNominalTag::sects>( j ).cast();
        const auto sectB = ( j == nPos - 1 )
                               ? sectA
                               : boundsarr[fState.offset()]
                                     .template get<LHCb::UT::TrackUtils::BoundariesNominalTag::sects>( j + 1 )
                                     .cast();

        assert( sectA != LHCb::UTDAQ::paddingSectorNumber && "sectA points to padding element" );
        assert( sectB != LHCb::UTDAQ::paddingSectorNumber && "sectB points to padding element" );
        assert( ( sectA > -1 ) && ( sectA < maxSectorNumber ) && "sector number out of bound" );
        assert( ( sectB > -1 ) && ( sectB < maxSectorNumber ) && "sector number out of bound" );

        // -- Sector is allowed to be a duplicate if it is the last element (as it has no consequence then)
        assert( ( ( sectA != sectB ) || ( j == nPos - 1 ) ) && "duplicated sectors" );

        const auto temp       = hitHandler.indices( sectA );
        const auto temp2      = hitHandler.indices( sectB );
        const auto firstIndex = temp.first;
        const auto shift      = temp2.first == temp.second;
        const auto lastIndex  = shift ? temp2.second : temp.second;
        j += shift;

        const auto myHs = myHits.simd();

        for ( auto i = firstIndex; i < lastIndex; i += simd::size ) {

          const auto mH = myHs[i];

          const auto yMin = min( mH.template get<UTHitsTag::yBegin>(), mH.template get<UTHitsTag::yEnd>() );
          const auto yMax = max( mH.template get<UTHitsTag::yBegin>(), mH.template get<UTHitsTag::yEnd>() );

          const auto yy    = stateY + ( mH.template get<UTHitsTag::zAtYEq0>() - stateZ ) * stateTy;
          const auto xx    = mH.template get<UTHitsTag::xAtYEq0>() + yy * mH.template get<UTHitsTag::dxDy>();
          const auto xPred = stateX + stateTx * ( mH.template get<UTHitsTag::zAtYEq0>() - stateZ ) +
                             bendParam * ( mH.template get<UTHitsTag::zAtYEq0>() - m_zUTField.value() );

          const auto absdx = abs( xx - xPred );

          if ( none( absdx < xTol ) ) continue;

          const auto mask =
              ( yMin - yTol < yy && yy < yMax + yTol ) && ( absdx < xTol ) && simd::loop_mask( i, lastIndex );

          if ( none( mask ) ) continue;

          const auto projDist = ( xPred - xx ) * ( m_zUTProj - m_zMSPoint ) /
                                ( mH.template get<UTHitsTag::zAtYEq0>() - m_zMSPoint.value() );

          auto muthit = hitsInLayers.compress_back<SIMDWrapper::InstructionSet::Best>( mask );
          muthit.template field<HitTag::zs>().set( mH.template get<UTHitsTag::zAtYEq0>() );
          muthit.template field<HitTag::sins>().set( mH.template get<UTHitsTag::cos>() *
                                                     -mH.template get<UTHitsTag::dxDy>() );
          muthit.template field<HitTag::weights>().set( mH.template get<UTHitsTag::weight>() );
          muthit.template field<HitTag::projections>().set( projDist );
          muthit.template field<HitTag::channelIDs>().set( mH.template get<UTHitsTag::channelID>() );
          muthit.template field<HitTag::indexs>().set( simd::indices( i ) ); // fill the index in the original hit
                                                                             // container
        }
      }
      nLayers += ( nSize != hitsInLayers.size() );
      hitsInLayers.layerIndices[layerIndex] = nSize;
      nSize                                 = hitsInLayers.size();
    }
    return nLayers > UTLayers;
  }
  //=========================================================================
  // Calculate Chi2
  //=========================================================================
  template <typename Container>
  float PrAddUTHitsTool::calculateChi2( float bestChi2, float p, unsigned bestSize, Container& goodUT,
                                        const UT::Mut::Hits& hitsInLayers ) const {

    // -- Fit a straight line to the points and calculate the chi2 of the hits with respect to the fitted track

    const auto xTol = m_xTol + m_xTolSlope / p;
    const auto fixW = m_fixedWeightFactor / ( xTol * xTol );
    // -- Fix slope by point at multiple scattering point
    const auto fixSlope = fixW * ( m_zUTProj - m_zMSPoint ) * ( m_zUTProj - m_zMSPoint );

    auto                 nDoF{ 0 };
    std::array<int, 4>   differentPlanes{};
    std::array<float, 6> mat{ fixW, 0.f, fixSlope, 0.f, 0.f, fixW };
    std::array<float, 3> rhs{};

    const auto& hitsInL = hitsInLayers.scalar();
    for ( auto iHit : goodUT ) {
      const auto uthit = hitsInL[iHit];
      const auto w     = uthit.weight().cast();
      const auto dz    = uthit.z().cast() - m_zUTProj;
      const auto t     = uthit.sin().cast();
      const auto dist2 = uthit.projection().cast();
      mat[0] += w;
      mat[1] += w * dz;
      mat[2] += w * dz * dz;
      mat[3] += w * t;
      mat[4] += w * dz * t;
      mat[5] += w * t * t;
      rhs[0] += w * dist2;
      rhs[1] += w * dist2 * dz;
      rhs[2] += w * dist2 * t;
      if ( !differentPlanes[uthit.planeCode().cast()]++ ) ++nDoF;
    }

    // -- Loop to remove outliers
    auto totChi2{ 0.f };
    while ( goodUT.size() >= m_minUThits ) {
      // -- This is needed since 'CholeskyDecomp' overwrites rhs
      // -- which is needed later on
      const auto saveRhs = std::array{ rhs[0], rhs[1], rhs[2] };

      CholeskyDecomp<float, 3> decomp( mat.data() );
      if ( !decomp ) return std::numeric_limits<float>::max();
      decomp.Solve( rhs );

      totChi2 = fixW * ( rhs[0] * rhs[0] + rhs[2] * rhs[2] +
                         ( m_zUTProj - m_zMSPoint ) * ( m_zUTProj - m_zMSPoint ) * rhs[1] * rhs[1] );
      auto       worst{ -1 };
      auto       worstChi2{ -1.f };
      const auto notMultiple = nDoF == static_cast<int>( goodUT.size() );
      for ( auto [idx, iHit] : LHCb::range::enumerate( goodUT ) ) {
        const auto  uthit = hitsInL[iHit];
        const float w     = uthit.weight().cast();
        const float dz    = uthit.z().cast() - m_zUTProj;
        const auto  dist  = uthit.projection().cast() - rhs[0] - rhs[1] * dz - rhs[2] * uthit.sin().cast();
        const auto  chi2  = w * dist * dist;
        totChi2 += chi2;
        if ( ( notMultiple || differentPlanes[uthit.planeCode().cast()] > 1 ) && worstChi2 < chi2 ) {
          worstChi2 = chi2;
          worst     = idx;
        }
      }

      if ( nDoF ) totChi2 /= nDoF;

      // -- Remove last point (outlier) if bad fit...or if nHits>8.
      // -- But also prefer 3 over 2 hits, not removing too aggressively 3/4 hits should be much more likely
      if ( goodUT.size() > MaxUTHits || ( bestChi2 < totChi2 && goodUT.size() > bestSize && goodUT.size() > 3 ) ) {
        const auto worstUT = hitsInL[goodUT[worst]];
        const auto w       = worstUT.weight().cast();
        const auto dz      = worstUT.z().cast() - m_zUTProj;
        const auto t       = worstUT.sin().cast();
        const auto dist2   = worstUT.projection().cast();
        mat[0] -= w;
        mat[1] -= w * dz;
        mat[2] -= w * dz * dz;
        mat[3] -= w * t;
        mat[4] -= w * dz * t;
        mat[5] -= w * t * t;
        rhs[0] = saveRhs[0] - w * dist2;
        rhs[1] = saveRhs[1] - w * dist2 * dz;
        rhs[2] = saveRhs[2] - w * dist2 * t;

        if ( 1 == differentPlanes[worstUT.planeCode().cast()]-- ) --nDoF;
        // remove the worst hit
        std::iter_swap( goodUT.nth( worst ), std::prev( goodUT.end() ) );
        goodUT.pop_back();
      } else {
        return totChi2;
      }
    }
    return totChi2;
  }
} // namespace LHCb::Pr
