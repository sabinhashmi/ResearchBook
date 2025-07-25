/***************************************************************************** \
* (c) Copyright 2000-2022 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

/*
  LoH:
  Change the removes for non-sorted tracks
  isValid is kept for the U/V hits as it is inefficient to rebuild the whole handler
  between parts, but U/V hit addition has some interplay between parts
*/

#include "DetDesc/GenericConditionAccessorHolder.h"
#include "Detector/FT/FTConstants.h"
#include "Event/FTCluster.h"
#include "Event/FTLiteCluster.h"
#include "Event/PrSciFiHits.h"
#include "Event/PrSeedTracks.h"
#include "Event/StateParameters.h"
#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "FTDAQ/FTInfo.h"
#include "FTDet/DeFTDetector.h"
#include "GaudiAlg/ISequencerTimerTool.h"
#include "GaudiKernel/StdArrayAsProperty.h"
#include "Kernel/STLExtensions.h" // ASSUME statement
#include "LHCbAlgs/Transformer.h"
#include "Magnet/DeMagnet.h"
#include "Math/CholeskyDecomp.h"
#include "PrHybridSeedTrack.h"
#include "PrKernel/FTGeometryCache.h"
#include "PrKernel/PrFTHitHandler.h"
#include "PrTrackFitterXYZ.h"
#include "PrTrackFitterXZ.h"
#include "PrTrackFitterYZ.h"
#include "TrackInterfaces/ITrackMomentumEstimate.h"
#include "/home/hashmi/ResearchBook/Tracking/Models/SciFiModel.cpp"
#include "HoughSearch.h"
namespace LHCb::Pr {
  namespace {
    constexpr unsigned int nParts = 2;
    static_assert( nParts <= 2 );
    using SmallDexes = std::vector<size_t>;
    using Tracks     = Seeding::Tracks;
    using SeedTag    = Seeding::Tag;
    using HitIter    = PrFTHitHandler<ModPrHit>::HitIter;
    using HitRevIter = PrFTHitHandler<ModPrHit>::HitRevIter;
    using ZoneCache  = Detector::FT::Cache::GeometryCache;
    struct SearchWindow {
      HitIter begin;
      HitIter end;
      float   xMinPrev{};
    };
    struct ZoneLimits {
      HitRevIter begin; // we always iterate from a point towards the beginning
      HitIter    end;   // we always iterate from a point towards the end
      float      lastHitX{};
    };
    using SearchWindowsX  = std::array<SearchWindow, LHCb::Detector::FT::nXLayersTotal>;
    using SearchWindowUV  = std::array<std::array<SearchWindow, 2>,
                                      LHCb::Detector::FT::nStations>; // BoundariesUV[U/V][T1,T2,T3][part]
    using SearchWindowsUV = std::array<SearchWindowUV, 2>;             // BoundariesUV[U/V][T1,T2,T3][part]

    using ZoneLimitsX = std::array<ZoneLimits, LHCb::Detector::FT::nXLayersTotal>;
    using ZoneLimitUV =
        std::array<std::array<ZoneLimits, 2>, LHCb::Detector::FT::nLayersTotal>; // BoundariesUV[U/V][T1,T2,T3]
    using ZoneLimitsUV = std::array<ZoneLimitUV, 2>;                             // BoundariesUV[U/V][T1,T2,T3][part]

    using TrackCandidates = std::array<std::vector<Hybrid::SeedTrack>, nParts>;
    using XCandidates     = std::array<std::vector<Hybrid::SeedTrackX>, nParts>;
    using TracksToRecover = std::array<std::vector<Hybrid::SeedTrackX>, nParts>;

    using SeedTrackHitsX          = Hybrid::SeedTrackHitsX;
    using SeedTrackHitsXIter      = Hybrid::SeedTrackHitsXIter;
    using SeedTrackHitsXConstIter = Hybrid::SeedTrackHitsXConstIter;
    using SeedTrackHits           = Hybrid::SeedTrackHits;
    using SeedTrackHitsIter       = Hybrid::SeedTrackHitsIter;
    using SeedTrackHitsConstIter  = Hybrid::SeedTrackHitsConstIter;
    static constexpr unsigned int maxParabolaHits =
        32; //---LoH: this is not a hard limit but serves as an estimate of the size
            // of the SeedTrackParabolaHits vector. It is practically never reached.
    using SeedTrackParabolaHits =
        std::vector<ModPrHit>; //---LoH: used to be a small vector of size maxParabolaHits. In reality,
                               //>99.99% of cases have size < 8. Could be exploited further.

    struct CommonXYHits {
      constexpr unsigned int operator()( unsigned int s1, unsigned int s2 ) {
        constexpr auto nCommonHits = std::array{ 7u, 6u, 6u, 5u, 4u }; // 70%*(12-nHits)
        auto           idx         = LHCb::Detector::FT::nLayersTotal - std::min( s1, s2 );
        assert( idx < nCommonHits.size() );
        return nCommonHits[idx];
      }
    };

    struct CommonXHits {
      constexpr unsigned int operator()( unsigned int s1, unsigned int s2 ) {
        constexpr auto nCommonHits = std::array{ 3u, 3u, 2u, 1u, 1u, 1u };
        auto           idx         = 2 * LHCb::Detector::FT::nXLayersTotal - s1 - s2;
        assert( idx < nCommonHits.size() );
        return nCommonHits[idx];
      }
    };

    struct CommonRecoverHits {
      constexpr unsigned int operator()( unsigned int s1, unsigned int s2 ) {
        constexpr auto nCommonHits = std::array{ 4u, 3u, 2u, 2u }; // 70%*(6-nHits)
        auto           idx         = LHCb::Detector::FT::nXLayersTotal - std::min( s1, s2 );
        assert( idx < nCommonHits.size() );
        return nCommonHits[idx];
      }
    };

    // Precalculated constants for a x-search in a given case
    struct CaseGeomInfoXZ {
      float                                                       invZf{};
      float                                                       invZlZf{};
      std::array<unsigned int, LHCb::Detector::FT::nXLayersTotal> zones{ {} };
      std::array<float, LHCb::Detector::FT::nXLayersTotal>        zLays{ {} };
      std::array<float, LHCb::Detector::FT::nXLayersTotal>        dzLays{ {} };
      std::array<float, LHCb::Detector::FT::nXLayersTotal>        dz2Lays{ {} };
      float                                                       t2Dist;
      //  float                                                   xInfCorr{};
      float txCorr{};
      float xRefCorr{};
      float delSeedCorr1{};
      float delSeedCorr2{};
      //  float                                                   delInfCorr;
    };

    // Parameters corresponding to a single two-hit combination in X
    struct TwoHitCombination {
      float tx{};
      float x0{};
      float xRef{};
      //    float invp{};
      //  float px{};
      //  float tx0{};
      float x0new{};
      float minPara{};
      float maxPara{};
    };

    // Physics constants
    constexpr float m_momentumScale = 35.31328;

    // Compiler option
    constexpr unsigned int maxXCandidates = 800;
    constexpr unsigned int maxCandidates  = 500;

    using namespace ranges;
    // Hit handling/ sorting etc...
    // Uses the reference to the PrHit
    struct compCoordXreverse {
      bool operator()( float lv, float rv ) const { return lv > rv; }
      bool operator()( float lv, const ModPrHit& rhs ) const { return ( *this )( lv, rhs.coord ); }
      bool operator()( const ModPrHit& lhs, const ModPrHit& rhs ) const { return ( *this )( lhs.coord, rhs.coord ); }
    };
    // Uses the coord method for faster access
    struct compCoordX {
      bool operator()( float lv, float rv ) const { return lv < rv; }
      bool operator()( float lv, const ModPrHit& rhs ) const { return ( *this )( lv, rhs.coord ); }
      bool operator()( const ModPrHit& lhs, const ModPrHit& rhs ) const { return ( *this )( lhs.coord, rhs.coord ); }
    };

    //---LoH: currently we cannot use coord as the flagging puts it to float::min -> breaks the logic of that function
    //---LoH: this function looks for the first hit whose x is smaller than the bound
    [[gnu::always_inline]] inline HitIter get_lowerBound_lin_reverse( HitRevIter low, HitIter high, float xMin ) {
      return std::find_if( std::reverse_iterator{ high }, low,
                           [xMin, cmp = compCoordXreverse{}]( const auto& i ) { return cmp( xMin, i ); } )
          .base();
    }
    // Version with a check (we could go to the end)
    [[gnu::always_inline]] inline HitIter get_lowerBound_lin( HitIter begin, HitIter end, float xMin ) {
      while ( begin != end && begin->coord < xMin ) { ++begin; }
      return begin;
    }

    // Version without check: we know we cannot go to the end
    [[gnu::always_inline]] inline HitIter get_lowerBound_lin( HitIter begin, float xMin ) {
      while ( xMin > begin->coord ) ++begin;
      return begin;
    }

    // search upperBound from known position to another known position (linear time)
    [[gnu::always_inline]] inline HitIter get_upperBound_lin( HitIter begin, HitIter end,
                                                              float xMax ) { //---LoH: 280k times in 100 events
      return std::find_if( begin, end, [xMax, cmp = compCoordX{}]( const auto& i ) { return cmp( xMax, i ); } );
    }

    // Update the bounds (Hiterator Pairs) and old Min => new Min searching linearly around current boundary begin
    [[gnu::always_inline]] inline void LookAroundMin( SearchWindow& searchWindow, float nMin,
                                                      const ZoneLimits& zoneLimits ) {
      if ( nMin < searchWindow.xMinPrev ) //---LoH: 113k times in 100 events
        searchWindow.begin = get_lowerBound_lin_reverse( zoneLimits.begin, searchWindow.begin, nMin );
      else {
        if ( nMin < zoneLimits.lastHitX )
          searchWindow.begin = get_lowerBound_lin( searchWindow.begin, nMin );
        else
          searchWindow.begin = get_lowerBound_lin( searchWindow.begin, zoneLimits.end, nMin );
      }
      searchWindow.xMinPrev = nMin;
    }

    void updateSmallDexes( SmallDexes& smallDexes, const PrFTHitHandler<ModPrHit>& hits ) noexcept {
      // reset values
      size_t iOffset( 0 );
      for ( unsigned int iZone = 0; iZone < LHCb::Detector::FT::nZonesTotal; iZone++ ) {
        for ( const auto& hit : hits.hits( iZone ) ) {
          smallDexes[hit.fullDex] = iOffset;
          iOffset++;
        }
      }
    }

    PrFTHitHandler<ModPrHit> makeHitHandler( const FT::Hits& hits ) noexcept {
      // Construct hit handler of ModPrHits
      // The track candidates will contain copies of the ModHits, but each will contain their
      // own index in the hit container, which can be used used to flag the original hits.
      PrFTHitHandler<ModPrHit> hitHandler( hits.size() );
      for ( unsigned int iZone = 0; iZone < LHCb::Detector::FT::nZonesTotal; iZone++ ) {
        const auto [begIndex, endIndex] = hits.getZoneIndices( iZone );
        for ( auto i = begIndex; i < endIndex; i++ ) { hitHandler.addHitInZone( iZone, hits.x( i ), i ); }
      }
      hitHandler.setOffsets();
      return hitHandler;
    }

    PrFTHitHandler<ModPrHit> consolidateHitHandler( const PrFTHitHandler<ModPrHit>& hits ) noexcept {
      // Returns a new hit handler where any invalid hit has been removed
      PrFTHitHandler<ModPrHit> hitHandler( hits.hits().size() );
      for ( unsigned int iZone = 0; iZone < LHCb::Detector::FT::nZonesTotal; iZone++ ) {
        for ( const auto& hit : hits.hits( iZone ) )
          if ( hit.isValid() ) hitHandler.addHitInZone( iZone, hit.coord, hit.fullDex );
      }
      hitHandler.setOffsets();
      return hitHandler;
    }

    [[gnu::always_inline]] inline bool findParabolaHits( float xMin, float xMax, const ZoneLimits& zoneLimits,
                                                         SearchWindow&          searchWindow,
                                                         SeedTrackParabolaHits& parabolaSeedHits ) noexcept {
      //---LoH: Called 6M times in 100 events
      parabolaSeedHits.clear();
      if ( xMin > zoneLimits.lastHitX ) return false; // no hit in T2x1 can correspond
      searchWindow.begin = get_lowerBound_lin( searchWindow.begin, xMin );
      if ( xMax < zoneLimits.lastHitX ) {
        for ( searchWindow.end = searchWindow.begin; searchWindow.end->coord <= xMax; ++searchWindow.end ) {
          parabolaSeedHits.push_back( *( searchWindow.end ) );
        }
      } else {
        for ( searchWindow.end = searchWindow.begin; searchWindow.end != zoneLimits.end; ++searchWindow.end ) {
          if ( searchWindow.end->coord > xMax ) break;
          parabolaSeedHits.push_back( *( searchWindow.end ) );
        }
      }
      return ( parabolaSeedHits.size() != 0 );
    }

  } // namespace

  /** @class HybridSeeding PrHybridSeeding.h
   *  Stand-alone seeding for the FT T stations
   *
   *  - OutputName : Name of the output container for the seed tracks. By Default it's LHCb::TrackLocation::Seed
   *  - NCases : Number of Cases for the algorithm ( value must be <=3 )
   *  - SlopeCorr : False by default. It change the errors on the hits from 1./err => 1./(err*cos(tx)*cos(tx))
   *  - maxNbestCluster[Case]: Amount of first N best clusters to process for each Case
   *  - MaxNHits : Force the algorithm to find tracks with at maximum N Hits ( N hits = N layers )
   *  - RemoveClones : Flag that allow to run the global clones removal ( true by default )
   *  - minNCommonUV : Number of common hits in the global clone removal step
   *  - RemoveClonesX : Flag that allow to run the intermediate clone killing for x-z projection
   *  - FlagHits : Flag that allow to flag the hits on track found by each Case
   *  - RemoveFlagged : If set to true, whatever flagged hits found by the Case-i is not used in Case-j , where j>i
   *  - UseCubicCorrection: Modify the track model for x(z) = ax+bx*dz + cx*dz*dz*(1+dRatio*dz)
   *  - dRatio : dRatio value
   *  - CConst : Constant to compute the backward projection x(0) ~ a_x - b_x * zref + cx * C ;
                 C = Integral (0, zRef) Integral (0, z) By(z') * dz * dz'
   *  - UseCorrPosition : Correct the position for the simultaneous fit using the yOnTrack shift , i.e. z(y) instead of
   z(0);
   *  - SizeToFlag[Case] : Tracks with NHits>=SizeToFlag will have the hits flagged
   *  - Flag_MaxChi2DoF_11Hits[Case] : If Hits<12 Flag only hits on track having Chi2DoF<Flag_MaxChi2DoF_11Hits[Case]
   *  - Flag_MaxX0_11Hits[Case] : If Hits<12 Flag only hits on track having |x0(backProjection)| <
   Flag_MaxX0_11Hits[Case]
   *
   *  Parameters x-z projection search
   *
   *  - 2-hit combination from T1-x + T3-x : given txinf = xT1/ZT1
   *  - L0_AlphaCorr[Case]  : Rotation angle obtained looking to txinf vs ( Delta ) , where Delta = xT3(True) -
   xT1+txinf(zT3-zT1).
   *  - L0_tolHp[Case]      : After rotating
   *                          Delta' =  Delta + L0_alphaCorrd[Case] vs
   *                          tx_inf, L0_alphaCorr[Case]*txinf - L0_tolHp[Case] <  xT3  <  L0_alphaCorr[Case]*txinf +
   L0_tolHp[Case]
   *  - 3-hit combination given straight line joining T1X and T3X. x0 is the straight line prediction
   *    from the two picked hits at z=0.
   *  - tx_picked = (xT3-xT1)/(zT3-zT1). x0 = xT1 - zT1 * tx_picked
   *  - xPredT2 = x0 + zT2 * tx_picked ( linear prediction from the 2 hit combination )
   *  - x0Corr[Case] : defines a new xPredT2' = xPredT2 + x0*x0Corr[Case]
   *                   ( rotation in the plane (xTrue - xPredT2) vs x0 to allign for the tolerances.
   *
   *  Considering only x0>0 ( equal by symmetry  for x0 <0)
   *  ( see https://indico.cern.ch/event/455022/contribution/2/attachments/1186203/1719828/main.pdf for reference )
   *  - X0SlopeChange[Case] : value of x0 at which start to open a larger upper tolerance ( max )
   *  - z0SlopeChangeDown[Case] : value of x0 at which start to open a larger lower toleance ( min )
   *  - ToleranceX0Up[Case] : upper tolerance when x0< X0SlopeChange[Case]
   *  - ToleranceX0Down[Case] : lower tolerance when x0 < X0SlopeChangeDown[Case]
   *  - x0Cut[Case] : Value of X0 where to define the new tolerance up and down
   *                  ( which then implicitely imply the opening tolerance up to X0SlopeChange(Down)[Case].
   *                    Must be > X0SlopeChange(Down) )
   *  - TolAtX0CutOpp[Case] : lower tolerance for the deviation to collect hits in T2 at x0Cut from the xPredT2'
   *  - TolAtX0Cut[Case] : upper tolerance for the deviation to collect hits in T2 at x0Cut from the xPredT2'
   *  - maxParabolaSeedHits : max number of hits to process collected in T2 given a 2 hit combination.
   *
   *  Collect remaining layers once a 3 hit combination is formed
   *  - TolXRemainign[Case] : from 3 hit the parabola ( + cubic correction ) is computed and the remaining
   *                          xlayers hits are collected if  the hits are found in within
   *                          TolXRemaining ( | hit::x(0 ) - xPred | < tolXPremaining )
   *
   *  Track is fitted in this scheme:
   *  - maxChi2HitsX[Case] : if Max Chi2(Hit) > maxChi2HitsX[Case] fit is failed and the outliers removed down to
   m_minXPlanes hits
   *  - maxChi2DoFX[Case] : max value of the Chi2 per DoF of the xz projections for each Case
   *
   *  UV Search
   *
   *  Collect compatible hits in UV layers:
   *  - HoleShape  : NoHole (do not account for the hole), Circular (see HoleRadius), or Rectangular (see HoleWidthX)
   *  - HoleRadius : remove hits found to have sqrt( x*x + y*y ) < HoleRadius (Circular geometry)
   *  - HoleWidthX : remove hits found to have (abs(x) < HoleWidthX and abs(y) < HoleWidthY) (Rectangular geometry)
   *  - HoleWidthY : see HoleWidthX
   *  - Positive defined tolerances for the y search  ( swapped when looking to lower module )
   *  - yMin : yMin Value to collect compatible hits in stereo
   *  - yMax : yMax Value to collect compatible hits in stereo
   *           ( upper track search in upper modyules : yMin < y < yMax )
   *  - yMin_TrFix : y Min Value to collect compatible hits in stereo when triangle fix is on
   *  - yMax_TrFix : y Max Value to collect compatible hits in stereo when triangle fix is on
   *                 ( upper track search in lower modules yMin_TrFix < t < yMax_TrFix
   *  - DoAsym : do asymmetric hit search taking into account stereo layers
   *  - TriangleFix : use triangle fixing
   *  - TriangleFix2ndOrder : use the info in Hit::yMax and Hit::yMin to remove the hits in upper modules in upper track
   leaking to y<0
   *
   *  Select hough cluster
   *  - minUV6[Case] : minUVLayers when XZ candidate has 6 hits  (planes)
   *  - minUV5[Case] : minUVLayers when XZ candidate has 5 hits  (planes)
   *  - minUV4[Case] : minUVLayers when XZ candidate has 4 hits  (planes)
   *  - minTot[Case] : remove outliers until reaching minToT[Case] hits
   *  - Hough like cluster selection : select cluster ( sorted by |y/z| )
   *
   *  Simultaneously fitting of the tracks
   *  - m_maxYAt0Low[Case] : If N Layers < 11: kill tracks having y(z=0) > m_maxYAt0Low[Case]
   *  - m_maxYAt0High[Case] : If N Layers >= 11: kill tracks having y(z=0) > m_maxYAt0Low[Case]
   *  - m_maxYAtRefLow[Case] : If N Layers < 11 : kill tracks having y(zRef)> maxYatzRefLow [Case]
   *  - m_maxYAtRefHigh[Case] : If N Layers >= 11 : kill tracks having y(zRef)> maxYatzRefLow [Case]
   *
   *  @author Renato Quagliani (rquaglia@cern.ch)
   *  @author Louis Henry (louis.henry@cern.ch)
   *  @author Salvatore Aiola (salvatore.aiola@cern.ch)
   *  @date   2020-03-27
   */

  class HybridSeeding
      : public Algorithm::Transformer<Seeding::Tracks( const FT::Hits&, ZoneCache const&, DeMagnet const& ),
                                      Algorithm::Traits::usesConditions<ZoneCache, DeMagnet>> {

  public:
    /// Standard constructor
    HybridSeeding( const std::string& name, ISvcLocator* pSvcLocator );
    /// initialization
    StatusCode initialize() override;
    /// main execution
    Tracks operator()( const FT::Hits&, ZoneCache const&, DeMagnet const& ) const override final;

    // Unlikely to change, still kept visible here
    static constexpr int  NCases      = 3;
    static constexpr bool Recover     = true;
    static constexpr int  RecoverCase = NCases;

  private:
  float minMaxScale(float value, float min, float max) const{
    return (value - min) / (max - min);
    }
    // Counters about the total number of tracks
    using SC = Gaudi::Accumulators::StatCounter<>;
    mutable SC                         m_outputTracksCnt{ this, "Created seed tracks" };
    mutable std::array<SC, 2>          m_outputTracksCnt_part{ SC{ this, "Created seed tracks (part 0)" },
                                                      SC{ this, "Created seed tracks (part 1)" } };
    mutable std::array<SC, NCases + 1> m_outputTracksCnt_cases{
        SC{ this, "Created seed tracks in case 0" }, SC{ this, "Created seed tracks in case 1" },
        SC{ this, "Created seed tracks in case 2" }, SC{ this, "Created seed tracks in recovery step" } };
    mutable std::array<SC, 2>          m_outputXZTracksCnt_part{ SC{ this, "Created XZ tracks (part 0)" },
                                                        SC{ this, "Created XZ tracks (part 1)" } };
    mutable std::array<SC, NCases + 1> m_outputXZTracksCnt_cases{
        SC{ this, "Created XZ tracks in case 0" }, SC{ this, "Created XZ tracks in case 1" },
        SC{ this, "Created XZ tracks in case 2" }, SC{ this, "Created XZ tracks in recovery step" } };

    // Internal counters - must be called with a buffer
    mutable std::array<SC, NCases> m_twoHitCombCnt_cases{ SC{ this, "Created two-hit combinations in case 0" },
                                                          SC{ this, "Created two-hit combinations in case 1" },
                                                          SC{ this, "Created two-hit combinations in case 2" } };
    mutable std::array<SC, NCases> m_threeHitCombCnt_cases{
        SC{ this, "Created T2x1 three-hit combinations in case 0" },
        SC{ this, "Created T2x1 three-hit combinations in case 1" },
        SC{ this, "Created T2x1 three-hit combinations in case 2" } };
    mutable std::array<SC, NCases> m_fullHitCombCnt_cases{ SC{ this, "Created full hit combinations in case 0" },
                                                           SC{ this, "Created full hit combinations in case 1" },
                                                           SC{ this, "Created full hit combinations in case 2" } };

    /** @brief Collect Hits in X layers producing the xz projections
     * @param part (if 1, y<0 ; if 0 , y>0)
     */
    void findXProjections( const FT::Hits& sciFiHits, ZoneCache const& zoneCache, unsigned int part, unsigned int iCase,
                           const PrFTHitHandler<ModPrHit>& hitHandler, XCandidates& xCandidates ) const noexcept;
    /** @brief Initialises parameters for X-hit finding
     */
    CaseGeomInfoXZ initializeXProjections( ZoneCache const& zoneCache, unsigned int iCase,
                                           unsigned int part ) const noexcept;

    /** @brief Updates parameters for each XZ pair
     */
    [[gnu::always_inline]] inline void updateXZCombinationPars( unsigned int iCase, float xFirst, float xLast,
                                                                const CaseGeomInfoXZ& xZones, float slope,
                                                                float slopeopp, float accTerm1, float accTerm2,
                                                                TwoHitCombination& hitComb ) const noexcept;

    /** @brief Fills hits from the first parabola layer
     */
    template <typename CountingBuffer>
    [[gnu::always_inline]] inline void
    fillXhits0( const FT::Hits& sciFiHits, unsigned int iCase, ModPrHitConstIter Fhit, const ModPrHit& Phit,
                ModPrHitConstIter Lhit, const CaseGeomInfoXZ& xZones, const TwoHitCombination& hitComb,
                const ZoneLimitsX& zoneLimits, std::vector<Pr::Hybrid::SeedTrackX>& xCandidates,
                SearchWindowsX& searchWindows, CountingBuffer& ) const noexcept;

    /** @brief Fills hits from the second parabola layer
     */
    template <typename CountingBuffer>
    [[gnu::always_inline]] inline void
    fillXhits1( const FT::Hits& sciFiHits, unsigned int iCase, ModPrHitConstIter Fhit, const ModPrHit& Phit,
                ModPrHitConstIter Lhit, const CaseGeomInfoXZ& xZones, const TwoHitCombination& hitComb,
                const ZoneLimitsX& zoneLimits, std::vector<Pr::Hybrid::SeedTrackX>& xCandidates,
                SearchWindowsX& searchWindows, CountingBuffer& ) const noexcept;

    /** @brief Creates a X track from xHits
     */
    void createXTrack( const FT::Hits& sciFiHits, unsigned int iCase, std::vector<Pr::Hybrid::SeedTrackX>& xCandidates,
                       Pr::Hybrid::SeedTrackX& cand ) const noexcept;

    SeedTrackHitsConstIter findWorstX( const Pr::Hybrid::SeedTrackX& track, int iCase ) const noexcept;

    /** @brief Finds hit corresponding to the missing parabola layer
     */
    [[gnu::always_inline]] inline void fillXhitParabola( float tol, float xAtZ, const SearchWindow& searchWindow,
                                                         SeedTrackHitsX& xHits ) const noexcept;

    /** @brief Finds hit corresponding to remaining layers
     */
    [[gnu::always_inline]] inline void fillXhitRemaining( float tol, float xAtZ, SearchWindow& searchWindow,
                                                          const ZoneLimits& zoneLimit,
                                                          SeedTrackHitsX&   xHits ) const noexcept;

    /** @brief Track recovering routine for x/z discarded tracks
     */
    void RecoverTrack( const FT::Hits& sciFiHits, ZoneCache const& zoneCache, SmallDexes& smallDexes,
                       PrFTHitHandler<ModPrHit>& FTHitHandler, TrackCandidates& trackCandidates,
                       XCandidates& xCandidates, TracksToRecover& tracksToRecover ) const noexcept;

    /** @brief Flag Hits on found tracks
     */
    void flagHits( unsigned int icase, unsigned int part, TrackCandidates& trackCandidates, unsigned int firstTrackCand,
                   const FT::Hits& sciFiHits, const SmallDexes& smallDexes,
                   PrFTHitHandler<ModPrHit>& hitHandler ) const noexcept;

    /** @brief Transform the tracks from the internal representation into Tracks
     *  @param tracks The tracks to transform
     */
    void makeLHCbTracks( const FT::Hits& sciFiHits, Tracks& result, unsigned int part,
                         const TrackCandidates& trackCandidates, const DeMagnet& magnet ) const noexcept;

  private:
    //  // Stereo layers
    std::array<std::array<unsigned int, 3>, 4> m_minUV; // m_minUV[case][6-nHits]
    std::array<std::array<float, nParts>, 2>   m_yMins;
    std::array<std::array<float, nParts>, 2>   m_yMaxs;

    //------------- Track recovering routine specific parameters
    Gaudi::Property<std::array<int, NCases>> m_nusedthreshold{ this, "nUsedThreshold", { 3, 2, 1 } };

    //------------- Momentum tuning
    Gaudi::Property<float> m_minP{ this, "MinP", 0.f * Gaudi::Units::MeV };
    Gaudi::Property<float> m_pFromTwoHitP1{ this, "PfromTwoHit_P1", 1.6322e-07f };
    Gaudi::Property<float> m_pFromTwoHitP2{ this, "PfromTwoHit_P2", -5.0217e-12f };

    //------------- Global configuration of the algorithm
    Gaudi::Property<unsigned int> m_minXPlanes{ this, "MinXPlanes", 4 };
    Gaudi::Property<bool>         m_removeClonesX{ this, "RemoveClonesX", true };
    Gaudi::Property<bool>         m_removeClones_forLead{ this, "RemoveClones_forLead", false };
    Gaudi::Property<float>        m_removeClones_sorted_distance{ this, "RemoveClones_sorted_distance", 100.f };
    Gaudi::Property<float>        m_removeClones_distance{ this, "RemoveClones_distance", 2.f };
    Gaudi::Property<float>        m_removeClones_distance_recover{ this, "RemoveClones_distance_recover", 5.f };

    //------------- Guesses about vector size
    Gaudi::Property<int> m_nTrackCandidates{ this, "nTrackCandidates", maxCandidates };
    Gaudi::Property<int> m_nXTrackCandidates{ this, "nXTrackCandidates", maxXCandidates };

    //------------ X-Z projections search parametrisation
    //=== 1st (T1) / Last (T3) Layer search windows
    Gaudi::Property<std::array<float, NCases>> m_alphaCorrection{ this, "L0_AlphaCorr", { 120.64, 510.64, 730.64 } };
    Gaudi::Property<std::array<float, NCases>> m_TolFirstLast{ this, "L0_tolHp", { 280.0, 540.0, 1080.0 } };

    //=== Add of the third hit in middle layers (p and Pt dependent, i.e., case dependent)
    Gaudi::Property<unsigned int> m_maxParabolaSeedHits{ this, "maxParabolaSeedHits", 8 };
    Gaudi::Property<bool>         m_parabolaSeedParabolicModel{ this, "ParabolaSeedParabolicModel", false };
    Gaudi::Property<std::array<float, NCases>> m_x0Cut{ this, "x0Cut", { 1500., 4000., 6000. } };
    Gaudi::Property<std::array<float, NCases>> m_x0Corr{ this, "x0Corr", { 1.002152, 1.001534, 1.001834 } };

    // In case we use the linear model
    Gaudi::Property<std::array<float, NCases>> m_x0SlopeChange{ this, "X0SlopeChange", { 400., 500., 500. } };
    Gaudi::Property<std::array<float, NCases>> m_tolX0SameSign{ this, "ToleranceX0Up", { 0.75, 0.75, 0.75 } };

    Gaudi::Property<std::array<float, NCases>> m_tolAtX0Cut{ this, "TolAtX0Cut", { 4.5, 8.0, 14.0 } };
    Gaudi::Property<std::array<float, NCases>> m_tolX0OppSign{ this, "ToleranceX0Down", { 0.75, 0.75, 0.75 } };
    Gaudi::Property<std::array<float, NCases>> m_x0SlopeChange2{ this, "X0SlopeChangeDown", { 2000., 2000., 2000. } };
    Gaudi::Property<std::array<float, NCases>> m_tolAtx0CutOppSign{ this, "TolAtX0CutOpp", { 0.75, 2.0, 7.0 } };

    //=== Add Hits in remaining X Layers after parabolic shape is found
    Gaudi::Property<std::array<float, NCases>> m_tolRemaining{ this, "TolXRemaining", { 1.0, 1.0, 1.0 } };

    //----------- Track Model parameters
    Gaudi::Property<float>                m_dRatio{ this, "dRatio", -0.000262 };
    Gaudi::Property<std::array<float, 3>> m_dRatioPar{ this, "dRatioPar", { 0.000267957, -8.651e-06, 4.60324e-05 } };
    Gaudi::Property<float> m_ConstC{ this, "CConst", 2.458e8 }; // Const value to compute the backward projection

    //----------- Beam hole parameters
    Gaudi::Property<bool>  m_removeBeamHole{ this, "RemoveBeamHole", false };
    Gaudi::Property<float> m_beamHoleX{ this, "BeamHoleX", 130. * Gaudi::Units::mm };
    Gaudi::Property<float> m_beamHoleY{ this, "BeamHoleY", 115. * Gaudi::Units::mm };

    //----------- Fit X/Z projection tolerances
    Gaudi::Property<std::array<float, NCases + 1>> m_maxChi2HitsX{ this, "maxChi2HitsX", { 5.5, 5.5, 5.5 } };
    Gaudi::Property<std::array<float, NCases + 1>> m_maxChi2DoFX{ this, "maxChi2DoFX", { 4.0, 5.0, 6.0 } };

    //----------- Full fit tolerances in standard cases and in recover case (index = 3)
    Gaudi::Property<std::array<float, NCases + 1>> m_minChi2HitFullRemove{
        this, "MinChi2HitFullRemove", { 6., 10., 12., 10. } };
    Gaudi::Property<std::array<float, NCases + 1>> m_minChi2PerDofFullRemove{
        this, "MinChi2PerDofFullRemove", { 1.75, 2., 1.5, 5. } };
    Gaudi::Property<std::array<float, NCases + 1>> m_maxChi2PerDofFullLow{
        this, "MaxChi2PerDofFullLow", { 1.75, 1.4, 1.25, 3.5 } };
    Gaudi::Property<std::array<float, NCases + 1>> m_maxYAt0Low{
        this, "MaxYAt0Low", { 9999999., 9999999., 120., 9999999. } };
    Gaudi::Property<std::array<float, NCases + 1>> m_minYAtRefLow{ this, "MinYAtRefLow", { 25., -1., 30., -1. } };
    Gaudi::Property<std::array<float, NCases + 1>> m_maxYAtRefLow{
        this, "MaxYAtRefLow", { 1080., 1000., 400., 2100. } };
    Gaudi::Property<std::array<float, NCases + 1>> m_minYAtRefHigh{ this, "MinYAtRefHigh", { -1., -1., -1., -1. } };
    Gaudi::Property<std::array<float, NCases + 1>> m_maxYAtRefHigh{
        this, "MaxYAtRefHigh", { 1200., 2100., 1600., 2200. } };

    //----------- UV-hits search parameters
    Gaudi::Property<float> m_yMin{ this, "yMin", -1.0 * Gaudi::Units::mm,
                                   [this]( const auto& ) {
                                     m_yMins[0][0] = m_yMin;
                                     m_yMaxs[1][1] = -m_yMin;
                                   },
                                   Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };
    Gaudi::Property<float> m_yMax{ this, "yMax", +2700. * Gaudi::Units::mm,
                                   [this]( const auto& ) {
                                     m_yMins[1][1] = -m_yMax;
                                     m_yMaxs[0][0] = m_yMax;
                                   },
                                   Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };
    Gaudi::Property<float> m_yMin_TrFix{ this, "yMin_TrFix", -2.0 * Gaudi::Units::mm,
                                         [this]( const auto& ) {
                                           m_yMins[0][1] = m_yMin_TrFix;
                                           m_yMaxs[1][0] = -m_yMin_TrFix;
                                         },
                                         Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };
    Gaudi::Property<float> m_yMax_TrFix{ this, "yMax_TrFix", +30.0 * Gaudi::Units::mm,
                                         [this]( const auto& ) {
                                           m_yMins[1][0] = -m_yMax_TrFix;
                                           m_yMaxs[0][1] = m_yMax_TrFix;
                                         },
                                         Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };

    Gaudi::Property<std::array<float, NCases + 1>> m_YSlopeBinWidth{
        this, "YSlopeBinWidth", { 0.0020, 0.0030, 0.0040, 0.0080 } };

    // Y fit
    Gaudi::Property<std::array<float, NCases + 1>> m_minChi2PerDofYRemove{
        this, "MinChi2PerDofYRemove", { 16., 35., 50., 120. } };
    Gaudi::Property<std::array<float, NCases + 1>> m_maxChi2PerDofYLow{
        this, "MaxChi2PerDofYLow", { 10., 11., 3., 30. } };
    Gaudi::Property<std::array<float, NCases + 1>> m_maxChi2PerDofYHigh{
        this, "MaxChi2PerDofYHigh", { 14.5, 35., 34., 100. } };
    Gaudi::Property<std::array<unsigned int, NCases>> m_minUV6{
        this,
        "minUV6",
        { 4, 4, 4 },
        [this]( const auto& ) {
          for ( int i = 0; i < NCases; ++i ) m_minUV[i][0] = m_minUV6[i];
        },
        Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };
    Gaudi::Property<std::array<unsigned int, NCases>> m_minUV5{
        this,
        "minUV5",
        { 5, 5, 4 },
        [this]( const auto& ) {
          for ( int i = 0; i < NCases; ++i ) m_minUV[i][1] = m_minUV5[i];
        },
        Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };
    Gaudi::Property<std::array<unsigned int, NCases>> m_minUV4{
        this,
        "minUV4",
        { 6, 6, 5 },
        [this]( const auto& ) {
          m_minUV[0][2] = m_minUV4[0];
          m_minUV[1][2] = m_minUV4[1];
          m_minUV[2][2] = m_minUV4[2];
        },
        Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };
    Gaudi::Property<std::array<unsigned int, NCases>> m_recover_minUV{
        this,
        "Recover_minUV",
        { 4, 5, 5 },
        [this]( const auto& ) { std::copy( m_recover_minUV.begin(), m_recover_minUV.end(), m_minUV[3].begin() ); },
        Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };
    Gaudi::Property<std::array<unsigned int, NCases + 1>> m_minTot{
        this,
        "minTot",
        { 9, 9, 9, 9 },
        [this]( const auto& ) {
          if ( std::any_of( m_minTot.begin(), m_minTot.end(), []( unsigned i ) { return i < 9u; } ) ) {
            throw GaudiException( "Algorithm does not support fewer than 9 hits in total (due to addStereo)", name(),
                                  StatusCode::FAILURE );
          };
        },
        Gaudi::Details::Property::ImmediatelyInvokeHandler{ true } };

    // Flag Hits Settings
    Gaudi::Property<std::array<float, NCases>> m_MaxChi2Flag{ this, "Flag_MaxChi2DoF_11Hits", { 0.5, 1.0, 1.0 } };
    Gaudi::Property<std::array<float, NCases>> m_MaxX0Flag{ this, "Flag_MaxX0_11Hits", { 100., 8000., 200. } };

    PublicToolHandle<ITrackMomentumEstimate> m_momentumTool{ this, "MomentumToolName", "FastMomentumEstimate" };

    Gaudi::Property<std::array<unsigned int, NCases + 1>> m_maxNClusters{ this, "maxNbestCluster", { 2, 4, 4, 3 } };

    mutable PublicToolHandle<ISequencerTimerTool> m_timerTool{ this, "TimerTool", "",
                                                               "Do not use in combination with multi-threading." };
    int                                           m_timerIndex{};

  protected:
    using HoughSearch           = Hough::HoughSearch<ModPrHit const*, 2, 2, LHCb::Detector::FT::nUVLayersTotal, 100>;
    using HoughCandidates       = HoughSearch::result_type;
    using HoughCandidate        = HoughCandidates::value_type;
    using HoughCandidateHitIter = HoughCandidate::iterator;

    template <bool RECO>
    class addStereoBase {
    protected:
      addStereoBase()                                            = default;
      addStereoBase( const HybridSeeding::addStereoBase<RECO>& ) = delete;
      addStereoBase( HybridSeeding::addStereoBase<RECO>&& )      = delete;
      addStereoBase& operator=( const addStereoBase<RECO>& )     = delete;
      addStereoBase& operator=( addStereoBase<RECO>&& )          = delete;
    };

    template <int CASE>
    class addStereo final : private addStereoBase<CASE == RecoverCase> {
      /** @brief Add Hits from Stereo layers on top of the x-z projections found.
      The hough clusters are collection of hits having ~same y/z.
      */
    public:
      addStereo( const HybridSeeding& hybridSeeding )
        requires( CASE == RecoverCase );
      addStereo( const HybridSeeding& hybridSeeding, TracksToRecover& tracksToRecover )
        requires( CASE != RecoverCase );
      addStereo()                                        = delete;
      addStereo( const HybridSeeding::addStereo<CASE>& ) = delete;
      addStereo( HybridSeeding::addStereo<CASE>&& )      = delete;
      template <int = CASE>
      addStereo( HybridSeeding&& ) = delete;

      addStereo& operator=( const addStereo<CASE>& ) = delete;
      addStereo& operator=( addStereo<CASE>&& )      = delete;

      inline void operator()( const FT::Hits& sciFiHits, const ZoneCache& zoneCache, unsigned int part,
                              const PrFTHitHandler<ModPrHit>& FTHitHandler, TrackCandidates& trackCandidates,
                              XCandidates& xCandidates ) const noexcept;

    private:
      /** @brief Creates the full tracks from the Hough clusters
       */
      inline bool createTracksFromHough( const FT::Hits& sciFiHits, std::vector<Pr::Hybrid::SeedTrack>& trackCandidates,
                                         const Pr::Hybrid::SeedTrackX& xCandidate, HoughCandidates& houghCand,
                                         size_t nHoughCand ) const noexcept;

      inline void createFullTrack( const FT::Hits& sciFiHits, std::vector<Pr::Hybrid::SeedTrack>& trackCandidates,
                                   const Pr::Hybrid::SeedTrackX& xCandidate, HoughCandidateHitIter itBeg,
                                   HoughCandidateHitIter itEnd, unsigned int& nTarget, float& chi2Target,
                                   unsigned int& nValidCandidates ) const noexcept;

      /** @brief Collect Hits in UV layers given the tolerances
       * @param xProje x-z plane track projection
       * @return HoughSearch object
       */
      inline HoughSearch initializeHoughSearch( const ZoneCache& zoneCache, unsigned int part,
                                                const Pr::Hybrid::SeedTrackX& xProje, const ZoneLimitsUV& borderZones,
                                                SearchWindowsUV& searchWindows ) const noexcept;

      template <int UV>
      inline void CollectLayerUV( ZoneCache const& zoneCache, unsigned int part, unsigned int layer,
                                  const Pr::Hybrid::SeedTrackX& xProje, const ZoneLimitUV& borderZones,
                                  SearchWindowUV& searchWindows, HoughSearch& hough ) const noexcept;

      inline void AddUVHit( unsigned int layer, ModPrHitConstIter it, float xPred, HoughSearch& hough ) const noexcept;

      inline void initializeUVsearchWindows( const PrFTHitHandler<ModPrHit>& FTHitHandler,
                                             SearchWindowsUV& searchWindows, ZoneLimitsUV& borderZones ) const noexcept;

      inline HoughCandidateHitIter findWorstY( const PrTrackFitterYZ& fitter, HoughCandidateHitIter& hitBegin,
                                               HoughCandidateHitIter& hitEnd ) const noexcept;

      inline SeedTrackHitsConstIter findWorstXY( const Pr::Hybrid::SeedTrack& track ) const noexcept;

    private:
      const HybridSeeding& m_hybridSeeding;
    };
  };

  // Declaration of the Algorithm Factory
  DECLARE_COMPONENT_WITH_ID( HybridSeeding, "PrHybridSeeding" )

  template <>
  class HybridSeeding::addStereoBase<false> {
  protected:
    addStereoBase( TracksToRecover& tracksToRecover ) : m_tracksToRecover( tracksToRecover ) {}
    addStereoBase()                                             = delete;
    addStereoBase( const HybridSeeding::addStereoBase<false>& ) = delete;
    addStereoBase( HybridSeeding::addStereoBase<false>&& )      = delete;
    addStereoBase& operator=( const addStereoBase<false>& )     = delete;
    addStereoBase& operator=( addStereoBase<false>&& )          = delete;

  protected:
    TracksToRecover& m_tracksToRecover;
  };

  // Clone removal namespace
  namespace {
    template <typename T, typename itHits>
    bool areClones( const T& tr2, itHits itBeg1, const itHits& itEnd1, unsigned int maxCommon ) {
      unsigned int nCommon = maxCommon;
      auto         itBeg2  = tr2.hits().begin();
      auto         itEnd2  = tr2.hits().end();
      while ( nCommon != 0 && itBeg1 != itEnd1 ) {
        if ( std::any_of( itBeg2, itEnd2, [id = itBeg1->fullDex]( const auto& h ) { return id == h.fullDex; } ) )
          --nCommon;
        ++itBeg1;
      }
      return ( nCommon == 0 );
    }

    /** @brief Given two tracks it checks if they are clones. Clone threshold is defined by the
        amount of shared hits ( clones if nCommon > maxCommon)
    */
    template <typename T>
    bool removeWorstTrack( T& t1, T& t2 ) {
      bool ret = !T::LowerBySize( t1, t2 );
      t1.setValid( !ret );
      t2.setValid( ret );
      return ret;
    }

    // Removes the worst of two tracks
    template <bool forLead, typename CommonHits, typename T>
    struct removeClonesT {
      void operator()( unsigned int part, T& candidates, const float minDist, CommonHits commonHits = CommonHits() ) {
        //---LoH: it is surely faster to do by a simpler alg
        size_t nCands = candidates[part].size();
        if ( nCands < 2 ) return;
        std::vector<float> xT1;
        std::vector<float> xT2;
        std::vector<float> xT3;
        if constexpr ( forLead ) {
          xT1.reserve( nCands );
          xT2.reserve( nCands );
          xT3.reserve( nCands );
          for ( auto itT1 = candidates[part].begin(); itT1 < candidates[part].end(); ++itT1 ) {
            xT1.push_back( itT1->xT1() );
            xT2.push_back( itT1->ax() );
            xT3.push_back( itT1->xT3() );
          }
        }
        for ( size_t i = 0; i < nCands - 1; i++ ) {
          auto& t1 = candidates[part][i];
          if ( !( t1.valid() ) ) continue;
          auto       itBeg1 = t1.hits().begin();
          const auto itEnd1 = t1.hits().end();
          for ( size_t j = i + 1; j < nCands; ++j ) {
            auto& t2 = candidates[part][j];
            if ( !t2.valid() ) continue;
            if constexpr ( forLead ) {
              if ( !( std::fabs( xT1[i] - xT1[j] ) < minDist || std::fabs( xT2[i] - xT2[j] ) < minDist ||
                      std::fabs( xT3[i] - xT3[j] ) < minDist ) )
                continue;
            } else {
              if ( !( std::fabs( t1.xT1() - t2.xT1() ) < minDist || std::fabs( t1.ax() - t2.ax() ) < minDist ||
                      std::fabs( t1.xT3() - t2.xT3() ) < minDist ) )
                continue;
            }
            if ( areClones( t2, itBeg1, itEnd1, commonHits( t1.size(), t2.size() ) ) ) {
              if ( removeWorstTrack( t1, t2 ) ) break;
            }
          }
        }
      }
    };

    auto removeClonesX_pp  = removeClonesT<false, CommonXHits, XCandidates>();
    auto removeClones_pp   = removeClonesT<false, CommonXYHits, TrackCandidates>();
    auto removeClones_lead = removeClonesT<true, CommonXYHits, TrackCandidates>();

    // Remove clones for lead
    void removeClonesX_sorted( unsigned int part, XCandidates& xCands, const float minDist, const float sortDist ) {
      size_t nCands = xCands[part].size();
      if ( nCands < 2 ) return;
      auto               endIt = xCands[part].end();
      std::vector<float> xT1;
      xT1.reserve( nCands );
      std::vector<float> xT2;
      xT2.reserve( nCands );
      std::vector<float> xT3;
      xT3.reserve( nCands );
      for ( auto itT1 = xCands[part].begin(); itT1 < endIt; ++itT1 ) {
        xT1.push_back( itT1->xT1() );
        xT2.push_back( itT1->ax() );
        xT3.push_back( itT1->xT3() );
      }

      //---LoH: it is surely faster to do by a simpler alg
      for ( size_t i = 0; i < nCands - 1; i++ ) {
        auto& t1 = xCands[part][i];
        if ( !t1.valid() ) continue;
        auto       itBeg1 = t1.hits().begin();
        const auto itEnd1 = t1.hits().end();
        const auto nHits1 = t1.size();
        auto       maxXT1 = xT1[i] + sortDist;
        for ( size_t j = i + 1; j < nCands; ++j ) {
          auto& t2 = xCands[part][j];
          if ( !t2.valid() ) continue;
          if ( xT1[j] > maxXT1 ) break;
          if ( !( std::abs( xT1[i] - xT1[j] ) < minDist || std::abs( xT2[i] - xT2[j] ) < minDist ||
                  std::abs( xT3[i] - xT3[j] ) < minDist ) )
            continue;
          if ( areClones( t2, itBeg1, itEnd1, CommonXHits()( nHits1, t2.size() ) ) ) {
            if ( removeWorstTrack( t1, t2 ) ) break;
          }
        }
      }
    }

    // Remove tracks to recover
    void removeClonesRecover( unsigned int part, TracksToRecover& tracksToRecover, XCandidates& xCandidates,
                              const float minDist, CommonRecoverHits commonHits = CommonRecoverHits() ) {
      size_t nCands = tracksToRecover[part].size();
      if ( nCands == 0 ) return;
      std::vector<float> xT1;
      xT1.reserve( nCands );
      std::vector<float> xT2;
      xT2.reserve( nCands );
      std::vector<float> xT3;
      xT3.reserve( nCands );
      for ( auto itT1 = tracksToRecover[part].begin(); itT1 < tracksToRecover[part].end(); ++itT1 ) {
        xT1.push_back( itT1->xT1() );
        xT2.push_back( itT1->ax() );
        xT3.push_back( itT1->xT3() );
      }

      for ( size_t i = 0; i < nCands; i++ ) {
        auto& t1 = tracksToRecover[part][i];
        if ( !t1.recovered() ) continue;
        auto       itBeg1 = t1.hits().begin();
        const auto itEnd1 = t1.hits().end();
        const auto nHits1 = t1.size();
        for ( size_t j = i + 1; j < nCands; ++j ) {
          auto& t2 = tracksToRecover[part][j];
          if ( !( t2.recovered() ) ) continue;
          if ( !( std::abs( xT1[i] - xT1[j] ) < minDist || std::abs( xT2[i] - xT2[j] ) < minDist ||
                  std::abs( xT3[i] - xT3[j] ) < minDist ) )
            continue;
          if ( areClones( t2, itBeg1, itEnd1, commonHits( nHits1, t2.size() ) ) ) {
            bool ret = !Pr::Hybrid::SeedTrackX::LowerBySize( t1, t2 );
            t1.setRecovered( !ret );
            t2.setRecovered( ret );
            if ( ret ) break;
          }
        }
        if ( t1.recovered() ) { xCandidates[int( part )].push_back( ( t1 ) ); }
      }
    }

    inline size_t nUsed( Pr::Hybrid::SeedTrackHitsConstIter itBeg, Pr::Hybrid::SeedTrackHitsConstIter itEnd,
                         const SmallDexes& smallDexes, const PrFTHitHandler<ModPrHit>& hitHandler ) {
      auto condition = [&hitHandler, &smallDexes]( const ModPrHit& h ) {
        return !hitHandler.hit( smallDexes[h.fullDex] ).isValid();
      };
      return std::count_if( itBeg, itEnd, condition );
    }

    template <typename Range>
    static bool hasT1T2T3Track( const FT::Hits& sciFiHits, const Range& hits ) noexcept {
      std::bitset<3> T{};
      for ( const auto& hit : hits ) {
        int planeBit = sciFiHits.planeCode( hit.fullDex ) / 4;
        ASSUME( planeBit < 3 );
        T[planeBit] = true;
        if ( T.all() ) return true;
      }
      return false;
    }

  } // namespace

  //=============================================================================
  // Standard constructor, initializes variable
  //=============================================================================

  HybridSeeding::HybridSeeding( const std::string& name, ISvcLocator* pSvcLocator )
      : Transformer( name, pSvcLocator,
                     { KeyValue{ "FTHitsLocation", PrFTInfo::SciFiHitsLocation },
                       KeyValue{ "ZoneCache", std::string{ ZoneCache::Location } + name },
                       KeyValue{ "Magnet", LHCb::Det::Magnet::det_path } },
                     KeyValue{ "OutputName", LHCb::TrackLocation::Seed } ) {}

  //=============================================================================
  // Initialization
  //=============================================================================
  StatusCode HybridSeeding::initialize() {
    return Transformer::initialize().andThen( [this] {
      if constexpr ( NCases > 3 ) {
        error() << "Algorithm does not support more than 3 Cases" << endmsg;
        return StatusCode::FAILURE;
      }
      if ( m_minTot[RecoverCase] < 9u ) {
        error() << "Algorithm does not support fewer than 9 hits in total (due to addStereo)" << endmsg;
        return StatusCode::FAILURE;
      }
      for ( unsigned int i = 0; i < NCases; ++i )
        if ( m_minTot[i] < 9 ) {
          error() << "Algorithm does not support fewer than 9 hits in total (due to addStereo)" << endmsg;
          return StatusCode::FAILURE;
        }
      // Zones cache, retrieved from the detector store
      // Zones cache
      this->template addConditionDerivation<ZoneCache( const DeFT& )>( { DeFTDetectorLocation::Default },
                                                                       this->template inputLocation<ZoneCache>() );
      if ( m_timerTool.isEnabled() ) m_timerIndex = m_timerTool->addTimer( this->name() );
      return StatusCode::SUCCESS;
    } );
  }

  //=============================================================================
  // Main execution
  //=============================================================================
  Tracks HybridSeeding::operator()( const FT::Hits& hits, ZoneCache const& zoneCache, DeMagnet const& magnet ) const {
    const auto scopedTimer = m_timerTool.get()->scopedTimer( m_timerIndex, m_timerTool.isEnabled() );
    // Containers
    TrackCandidates trackCandidates; //---LoH: probably could be changed to array(static_vector)
    XCandidates     xCandidates;     //---LoH: probably could be changed to static_vector
    TracksToRecover tracksToRecover; //---LoH: probably could be changed to static_vector

    trackCandidates[0].reserve( m_nTrackCandidates );
    trackCandidates[1].reserve( m_nTrackCandidates );
    xCandidates[0].reserve( m_nXTrackCandidates );
    xCandidates[1].reserve( m_nXTrackCandidates );
    tracksToRecover[0].reserve( m_nXTrackCandidates );
    tracksToRecover[1].reserve( m_nXTrackCandidates );

    //==========================================================
    // Hits are ready to be processed
    //==========================================================
    unsigned int             nHits         = hits.size();
    PrFTHitHandler<ModPrHit> oriHitHandler = makeHitHandler( hits );
    SmallDexes               oriSmallDexes( nHits, 0 );
    updateSmallDexes( oriSmallDexes, oriHitHandler );

    PrFTHitHandler<ModPrHit> tmpHitHandler;
    SmallDexes               smallDexes = oriSmallDexes;

    PrFTHitHandler<ModPrHit>* hitHandler = &oriHitHandler;
    //========================================================
    //------------------MAIN SEQUENCE IS HERE-----------------
    //========================================================
    auto main_loop = [&]( auto icase ) {
      //----- Loop through lower and upper half
      std::array<unsigned int, nParts> nTracksBefore;
      for ( unsigned int part = 0; nParts > part; ++part ) {
        nTracksBefore[part] = trackCandidates[part].size();
        xCandidates[part].clear(); // x candidates up cleaned every Case!
        findXProjections( hits, zoneCache, part, icase, *hitHandler, xCandidates );
        if ( m_removeClonesX ) {
          if ( m_removeClones_forLead )
            removeClonesX_sorted( part, xCandidates, m_removeClones_distance, m_removeClones_sorted_distance );
          else
            removeClonesX_pp( part, xCandidates, m_removeClones_distance );
        } //---LoH: this probably should be encoded in the logic. Clone tracks are close to each other.
        m_outputXZTracksCnt_part[part] += xCandidates[part].size();
        m_outputXZTracksCnt_cases[icase] += xCandidates[part].size();
        addStereo<icase> addStereoCase( *this, tracksToRecover );
        addStereoCase( hits, zoneCache, part, *hitHandler, trackCandidates, xCandidates );
        m_outputTracksCnt_cases[icase] += trackCandidates[part].size();
      }
      // Flag found Hits at the end of each single case ( exclude the latest one )
      if ( ( icase + 1 < NCases ) ) {
        for ( unsigned int part = 0; nParts > part; ++part )
          flagHits( icase, part, trackCandidates, nTracksBefore[part], hits, smallDexes, *hitHandler );
        tmpHitHandler = consolidateHitHandler( *hitHandler );
        hitHandler    = &tmpHitHandler;
        updateSmallDexes( smallDexes, *hitHandler );
      }
    };
    Utils::unwind<0, NCases>( main_loop );

    // Recovering step
    if ( Recover ) {
      xCandidates[0].clear();
      xCandidates[1].clear();
      int totCands = -( trackCandidates[0].size() + trackCandidates[1].size() );
      RecoverTrack( hits, zoneCache, oriSmallDexes, oriHitHandler, trackCandidates, xCandidates, tracksToRecover );
      m_outputTracksCnt_cases[NCases] += ( totCands + trackCandidates[0].size() + trackCandidates[1].size() );
    }

    // Clone removal ( up/down )
    for ( unsigned int part = 0; part < nParts; ++part ) {
      if ( m_removeClones_forLead )
        removeClones_lead( part, trackCandidates, m_removeClones_distance );
      else
        removeClones_pp( part, trackCandidates, m_removeClones_distance );
      m_outputTracksCnt_part[part] += trackCandidates[part].size();
    }

    Tracks result;
    result.reserve( trackCandidates[0].size() + trackCandidates[1].size() );
    // Convert LHCb tracks
    for ( unsigned int part = 0; part < nParts; ++part ) {
      makeLHCbTracks( hits, result, part, trackCandidates, magnet );
    }
    return result;
  }

  //
  //===========================================================================================================================
  // EVERYTHING RELATED TO UV-HIT ADDITION
  //===========================================================================================================================

  template <int CASE>
  HybridSeeding::addStereo<CASE>::addStereo( const HybridSeeding& hybridSeeding )
    requires( CASE == RecoverCase )
      : addStereoBase<true>(), m_hybridSeeding( hybridSeeding ) {}

  template <int CASE>
  HybridSeeding::addStereo<CASE>::addStereo( const HybridSeeding& hybridSeeding, TracksToRecover& tracksToRecover )
    requires( CASE != RecoverCase )
      : addStereoBase<false>( tracksToRecover ), m_hybridSeeding( hybridSeeding ) {}

  template <int CASE>
  void HybridSeeding::addStereo<CASE>::operator()( const FT::Hits& sciFiHits, const ZoneCache& zoneCache,
                                                   unsigned int part, const PrFTHitHandler<ModPrHit>& FTHitHandler,
                                                   TrackCandidates& trackCandidates,
                                                   XCandidates&     xCandidates ) const noexcept {
    // Initialise bounds
    SearchWindowsUV searchWindows;
    ZoneLimitsUV    zoneLimits;
    initializeUVsearchWindows( FTHitHandler, searchWindows, zoneLimits );

    //---LoH: Loop on the xCandidates
    //---LoH: They are not sorted by xProje and their order can vary
    for ( auto const& itT : xCandidates[part] ) {
      if ( !itT.valid() ) continue;
      auto            hough = initializeHoughSearch( zoneCache, part, itT, zoneLimits, searchWindows );
      HoughCandidates houghCand;
      bool            hasAdded      = false;
      auto            lastCandidate = hough.search( houghCand.begin() );
      int             n_cand        = lastCandidate - houghCand.begin();
      if ( n_cand > 0 ) {
        // you have a minimal number of total hits to find on track dependent if standard 3 cases or from Recover
        // routine
        hasAdded = createTracksFromHough( sciFiHits, trackCandidates[part], itT, houghCand, n_cand );
      }
      if constexpr ( CASE != RecoverCase ) {
        if ( !hasAdded ) this->m_tracksToRecover[part].push_back( itT );
      }
    }
  }

  //---LoH: Can be put outside of class
  template <int CASE>
  void HybridSeeding::addStereo<CASE>::initializeUVsearchWindows( const PrFTHitHandler<ModPrHit>& FTHitHandler,
                                                                  SearchWindowsUV&                searchWindows,
                                                                  ZoneLimitsUV& zoneLimits ) const noexcept {
    for ( unsigned int layer : { LHCb::Detector::FT::UpperZones::T1U, LHCb::Detector::FT::UpperZones::T2U,
                                 LHCb::Detector::FT::UpperZones::T3U, LHCb::Detector::FT::UpperZones::T1V,
                                 LHCb::Detector::FT::UpperZones::T2V, LHCb::Detector::FT::UpperZones::T3V } ) {
      auto uv       = ( ( layer + 1 ) / 2 ) % 2;
      auto iStation = layer / 8;
      for ( unsigned int iPart = 0; iPart < nParts; ++iPart ) {
        float lastHitX( -5000.f );
        // Fill the UV bounds
        auto r = FTHitHandler.hits( layer - iPart );
        if ( !r.empty() ) lastHitX = r.back().coord;
        searchWindows[uv][iStation][iPart] = { r.begin(), r.end(), std::numeric_limits<float>::lowest() };
        zoneLimits[uv][iStation][iPart]    = { std::reverse_iterator{ r.begin() }, r.end(), lastHitX };
      }
    }
  }

  //---LoH: Can be put outside of class if takes the seeding as argument.
  template <int CASE>
  HybridSeeding::HoughSearch HybridSeeding::addStereo<CASE>::initializeHoughSearch(
      const ZoneCache& zoneCache, unsigned int part, const Pr::Hybrid::SeedTrackX& xProje,
      const ZoneLimitsUV& zoneLimits, SearchWindowsUV& searchWindows ) const noexcept {

    auto calculate_factor = [&]( int zone ) -> float {
      const auto layer    = zone / Detector::FT::nHalfLayers;
      const auto side     = xProje.x( zoneCache.z( layer ) ) > 0.f ? LHCb::Detector::FTChannelID::Side::A
                                                                   : LHCb::Detector::FTChannelID::Side::C;
      const auto partSign = part < 1 ? 1.f : -1.f;
      return partSign / ( zoneCache.z( zone, side ) * zoneCache.dxdy( zone, side ) );
    };
    auto calculate_factors = [&]( auto... layers ) { return std::array{ calculate_factor( layers )... }; };

    auto hough = HoughSearch{
        2,
        -m_hybridSeeding.m_YSlopeBinWidth[CASE] / 2,
        m_hybridSeeding.m_YSlopeBinWidth[CASE],
        { calculate_factors( LHCb::Detector::FT::UpperZones::T1U, LHCb::Detector::FT::UpperZones::T1V,
                             LHCb::Detector::FT::UpperZones::T2U, LHCb::Detector::FT::UpperZones::T2V,
                             LHCb::Detector::FT::UpperZones::T3U, LHCb::Detector::FT::UpperZones::T3V ) } };
    // u-layers
    for ( unsigned int layer : { LHCb::Detector::FT::UpperZones::T1U, LHCb::Detector::FT::UpperZones::T2U,
                                 LHCb::Detector::FT::UpperZones::T3U } )
      this->CollectLayerUV<0>( zoneCache, part, layer, xProje, zoneLimits[0], searchWindows[0], hough );
    // v-layers
    for ( unsigned int layer : { LHCb::Detector::FT::UpperZones::T1V, LHCb::Detector::FT::UpperZones::T2V,
                                 LHCb::Detector::FT::UpperZones::T3V } )
      this->CollectLayerUV<1>( zoneCache, part, layer, xProje, zoneLimits[1], searchWindows[1], hough );
    return hough;
  }

  template <int CASE>
  template <int UV> // 0 if U, 1 if V
  void HybridSeeding::addStereo<CASE>::CollectLayerUV( const ZoneCache& zoneCache, unsigned int part, unsigned int zone,
                                                       const Pr::Hybrid::SeedTrackX& xProje,
                                                       const ZoneLimitUV& zoneLimits, SearchWindowUV& searchWindows,
                                                       HoughSearch& hough ) const noexcept {
    static_assert( UV == 0 || UV == 1, "UV must be 0 (U) or 1 (V)!" );
    const auto side  = xProje.x( zoneCache.z( zone / Detector::FT::nHalfLayers ) ) > 0.f
                           ? LHCb::Detector::FTChannelID::Side::A
                           : LHCb::Detector::FTChannelID::Side::C;
    const auto dxDy  = zoneCache.dxdy( zone, side );
    const auto xPred = xProje.xFromDz( zoneCache.z( zone, side ) - Pr::Hybrid::zReference );
    // Note: UV layers are 1, 3, 11, 13, 19, 21
    unsigned int iStation     = zone / ( Detector::FT::nLayers * Detector::FT::nHalfLayers );
    unsigned int iStereoLayer = zone / ( Detector::FT::nHalfLayers * 2 ); // only half of the layers are stereo

    // Part 0
    std::array<float, 2> yMinMax{ m_hybridSeeding.m_yMins[part][0], m_hybridSeeding.m_yMaxs[part][0] };
    // The 1 - UV expression will be optmized away by the compiler (const expression)
    std::array<float, 2> xMinMax{ xPred - yMinMax[1 - UV] * dxDy, xPred - yMinMax[UV] * dxDy };
    //---LoH: in most cases (~70%), xMinMax is larger than xMinPrev. This is the same proportion between U and V
    LookAroundMin( searchWindows[iStation][0], xMinMax[0], zoneLimits[iStation][0] );
    for ( searchWindows[iStation][0].end = searchWindows[iStation][0].begin;
          searchWindows[iStation][0].end != zoneLimits[iStation][0].end; ++searchWindows[iStation][0].end ) {
      if ( searchWindows[iStation][0].end->coord < xMinMax[0] ) continue;
      if ( searchWindows[iStation][0].end->coord > xMinMax[1] ) break;
      AddUVHit( iStereoLayer, searchWindows[iStation][0].end, xPred, hough );
    }
    // Part 1
    yMinMax = { m_hybridSeeding.m_yMins[part][1], m_hybridSeeding.m_yMaxs[part][1] };
    // The 1 - UV expression will be optmized away by the compiler (const expression)
    xMinMax = { xPred - yMinMax[1 - UV] * dxDy, xPred - yMinMax[UV] * dxDy };
    LookAroundMin( searchWindows[iStation][1], xMinMax[0], zoneLimits[iStation][1] );
    for ( searchWindows[iStation][1].end = searchWindows[iStation][1].begin;
          searchWindows[iStation][1].end != zoneLimits[iStation][1].end; ++searchWindows[iStation][1].end ) {
      if ( searchWindows[iStation][1].end->coord < xMinMax[0] ) continue;
      if ( searchWindows[iStation][1].end->coord > xMinMax[1] ) break;
      AddUVHit( iStereoLayer, searchWindows[iStation][1].end, xPred, hough );
    }
  }

  template <int CASE>
  void HybridSeeding::addStereo<CASE>::AddUVHit( unsigned int layer, ModPrHitConstIter it, float xPred,
                                                 HoughSearch& hough ) const noexcept {
    float y = xPred - it->coord;
    hough.add( layer, y, &( *it ) );
  }

  template <int CASE>
  void HybridSeeding::addStereo<CASE>::createFullTrack( const FT::Hits&                     sciFiHits,
                                                        std::vector<Pr::Hybrid::SeedTrack>& trackCandidates,
                                                        const Pr::Hybrid::SeedTrackX&       xCandidate,
                                                        HoughCandidateHitIter itBeg, HoughCandidateHitIter itEnd,
                                                        unsigned int& nTarget, float& chi2Target,
                                                        unsigned int& nValidCandidates ) const noexcept {
    constexpr int      LOW_NTOT_YSEL_TH = CASE == 2 ? 10 : 11;
    constexpr int      LOW_NTOT_TH      = 11;
    constexpr int      LOW_NUV_TH       = CASE == 2 ? 5 : 6;
    constexpr int      FITTER_XYZ_LOOPS = CASE == 0 ? 2 : 3;
    const unsigned int nXHits           = xCandidate.size();
    const unsigned int minUV            = m_hybridSeeding.m_minUV[CASE][LHCb::Detector::FT::nXLayersTotal - nXHits];

    unsigned int nUVHits = itEnd - itBeg;
    unsigned int nHits   = nXHits + nUVHits;

    if ( nHits < nTarget || nUVHits < minUV ) return;

    PrTrackFitterYZ fitterYZ( xCandidate );
    bool            fitY = fitterYZ.fit( sciFiHits, itBeg, itEnd );
    if ( !fitY ) return;

    auto worstHitY = findWorstY( fitterYZ, itBeg, itEnd );
    bool canRemoveHitY =
        nHits > nTarget && nUVHits > m_hybridSeeding.m_minUV[CASE][LHCb::Detector::FT::nXLayersTotal - nXHits];
    while ( worstHitY != itEnd ) {
      if ( !canRemoveHitY ) return;
      --itEnd;
      std::copy( worstHitY + 1, itEnd + 1, worstHitY );
      *itEnd = HoughSearch::invalid_value<std::remove_pointer_t<HoughCandidateHitIter>>();
      --nHits;
      --nUVHits;
      fitY = fitterYZ.fit( sciFiHits, itBeg, itEnd );
      if ( !fitY ) return;
      worstHitY = findWorstY( fitterYZ, itBeg, itEnd );
      canRemoveHitY =
          nHits > nTarget && nUVHits > m_hybridSeeding.m_minUV[CASE][LHCb::Detector::FT::nXLayersTotal - nXHits];
    }

    if ( ( nUVHits >= LOW_NUV_TH && fitterYZ.chi2PerDoF() > m_hybridSeeding.m_maxChi2PerDofYHigh[CASE] ) ||
         ( nUVHits < LOW_NUV_TH && fitterYZ.chi2PerDoF() > m_hybridSeeding.m_maxChi2PerDofYLow[CASE] ) )
      return;

    // Do the beam hole filter
    if ( m_hybridSeeding.m_removeBeamHole ) {
      if ( std::abs( fitterYZ.ay() ) < m_hybridSeeding.m_beamHoleY &&
           std::abs( xCandidate.ax() ) < m_hybridSeeding.m_beamHoleX )
        return;
      for ( auto dz :
            { StateParameters::ZBegT - Pr::Hybrid::zReference, StateParameters::ZEndT - Pr::Hybrid::zReference } ) {
        auto y = std::abs( fitterYZ.ay() + fitterYZ.by() * dz );
        if ( y < m_hybridSeeding.m_beamHoleY )
          if ( std::abs( xCandidate.xFromDz( dz ) ) < m_hybridSeeding.m_beamHoleX ) return;
      }
    }

    Pr::Hybrid::SeedTrack trackCand( xCandidate );
    for ( auto hit = itBeg; hit != itEnd; ++hit ) trackCand.hits().push_back( *( *hit ) );
    trackCand.setYParam( fitterYZ.ay(), fitterYZ.by() );

    PrTrackFitterXYZ<FITTER_XYZ_LOOPS> fitterXYZ;
    bool fitXY = fitterXYZ.fit( sciFiHits, trackCand, m_hybridSeeding.m_dRatioPar.value() );
    if ( !fitXY ) return;

    auto worstHit = findWorstXY( trackCand );
    while ( worstHit != trackCand.hits().cend() ) {
      if ( !canRemoveHitY ) return;
      if ( sciFiHits.planeCode( worstHit->fullDex ) % 4 == 1 || sciFiHits.planeCode( worstHit->fullDex ) % 4 == 2 ) {
        --nUVHits;
      }
      --nHits;
      trackCand.hits().erase( worstHit );

      fitXY = fitterXYZ.fit( sciFiHits, trackCand, m_hybridSeeding.m_dRatioPar.value() );
      if ( !fitXY ) return;
      worstHit      = findWorstXY( trackCand );
      canRemoveHitY = nHits > nTarget && nUVHits > m_hybridSeeding.m_minUV[CASE][6 - nHits + nUVHits];
    }

    if ( nHits > nTarget ) {
      if ( nHits < LOW_NTOT_TH && trackCand.chi2PerDoF() > m_hybridSeeding.m_maxChi2PerDofFullLow[CASE] ) return;
    } else {
      if ( trackCand.chi2PerDoF() > chi2Target ) return;
    }

    float absY0   = std::fabs( trackCand.y0() );
    float absYref = std::fabs( trackCand.yRef() );
    if ( nHits < LOW_NTOT_YSEL_TH ) {
      if ( absY0 >= m_hybridSeeding.m_maxYAt0Low[CASE] || absYref >= m_hybridSeeding.m_maxYAtRefLow[CASE] ||
           absYref < m_hybridSeeding.m_minYAtRefLow[CASE] )
        return;
    } else {
      if ( absYref >= m_hybridSeeding.m_maxYAtRefHigh[CASE] || absYref < m_hybridSeeding.m_minYAtRefHigh[CASE] ) return;
    }

    trackCand.setnXnY( nHits - nUVHits, nUVHits );
    trackCand.setXT1( trackCand.xFromDz( StateParameters::ZBegT - Pr::Hybrid::zReference ) );
    trackCand.setXT3( trackCand.xFromDz( StateParameters::ZEndT - Pr::Hybrid::zReference ) );

    if ( nValidCandidates == 0 ) {
      trackCandidates.push_back( trackCand );
    } else {
      trackCandidates.back() = trackCand;
    }
    ++nValidCandidates;
    nTarget    = trackCandidates.back().size();
    chi2Target = trackCandidates.back().chi2PerDoF();
  }

  template <int CASE>
  bool HybridSeeding::addStereo<CASE>::createTracksFromHough( const FT::Hits&                     sciFiHits,
                                                              std::vector<Pr::Hybrid::SeedTrack>& trackCandidates,
                                                              const Pr::Hybrid::SeedTrackX&       xCandidate,
                                                              HoughCandidates&                    houghCand,
                                                              size_t nHoughCand ) const noexcept {
    unsigned int nValidCandidates = 0;
    unsigned int nTarget          = m_hybridSeeding.m_minTot[CASE];
    float        chi2Target       = m_hybridSeeding.m_maxChi2PerDofFullLow[CASE];
    for ( size_t iCand = 0; iCand < nHoughCand; ++iCand ) {
      auto& hcand = houghCand[iCand];
      auto  itBeg = hcand.begin();
      auto  itEnd = hcand.end();
      while ( ( itEnd - itBeg ) > 4 && HoughSearch::invalid( *( itEnd - 1 ) ) ) --itEnd;
      if ( HoughSearch::invalid( *( itEnd - 1 ) ) ) continue;
      createFullTrack( sciFiHits, trackCandidates, xCandidate, itBeg, itEnd, nTarget, chi2Target, nValidCandidates );
    }
    return ( nValidCandidates != 0 );
  }

  template <int CASE>
  HybridSeeding::HoughCandidateHitIter
  HybridSeeding::addStereo<CASE>::findWorstY( const PrTrackFitterYZ& fitterYZ, HoughCandidateHitIter& hitBegin,
                                              HoughCandidateHitIter& hitEnd ) const noexcept {
    const auto worst_it =
        std::max_element( fitterYZ.chi2Hits().cbegin(), fitterYZ.chi2Hits().cbegin() + ( hitEnd - hitBegin ) );
    auto worst_pos = worst_it - fitterYZ.chi2Hits().cbegin();
    if ( fitterYZ.chi2PerDoF() < m_hybridSeeding.m_minChi2PerDofYRemove[CASE] ) {
      return hitEnd;
    } else {
      return hitBegin + worst_pos;
    }
  }

  template <int CASE>
  SeedTrackHitsConstIter
  HybridSeeding::addStereo<CASE>::findWorstXY( const Pr::Hybrid::SeedTrack& track ) const noexcept {
    const auto worst_it  = std::max_element( track.chi2Hits().cbegin(), track.chi2Hits().cbegin() + track.size() );
    auto       worst_pos = worst_it - track.chi2Hits().cbegin();
    if ( track.chi2PerDoF() < m_hybridSeeding.m_minChi2PerDofFullRemove[CASE] &&
         track.chi2Hits()[worst_pos] < m_hybridSeeding.m_minChi2HitFullRemove[CASE] ) {
      return track.hits().cend();
    } else {
      return track.hits().cbegin() + worst_pos;
    }
  }

  void HybridSeeding::RecoverTrack( const FT::Hits& sciFiHits, const ZoneCache& zoneCache, SmallDexes& smallDexes,
                                    PrFTHitHandler<ModPrHit>& FTHitHandler, TrackCandidates& trackCandidates,
                                    XCandidates& xCandidates, TracksToRecover& tracksToRecover ) const noexcept {
    // Flagging all hits in the track candidates
    for ( unsigned int part = 0; part < nParts; ++part ) {
      for ( auto const& itT1 : trackCandidates[part] ) {
        if ( !itT1.valid() ) continue;
        for ( auto const& hit : itT1.hits() ) {
          size_t smallDex = smallDexes[hit.fullDex];
          FTHitHandler.hit( smallDex ).setInvalid();
        }
      }
    }
    // Check for the nUsed
    for ( unsigned int part = 0; part < nParts; ++part ) {
      for ( auto&& itT1 : tracksToRecover[part] ) {
        int nUsed_threshold = m_nusedthreshold[6 - itT1.size()]; // LoH
        //---LoH: this should be asserted: size must be 4, 5 or 6
        //---LoH: this checks if we have enough valid hits.
        int nUsedHits = nUsed( itT1.hits().begin(), itT1.hits().end(), smallDexes, FTHitHandler );
        if ( nUsedHits < nUsed_threshold ) {
          if ( nUsedHits > 0 ) {
            while ( nUsedHits > 0 ) {
              auto it = std::remove_if( itT1.hits().begin(), itT1.hits().end(),
                                        [&smallDexes, &FTHitHandler]( const ModPrHit& hit ) {
                                          return !FTHitHandler.hit( smallDexes[hit.fullDex] ).isValid();
                                        } );
              auto n  = std::distance( it, itT1.hits().end() );
              itT1.hits().erase( it, itT1.hits().end() );
              nUsedHits -= n;
            }
            if ( itT1.hits().size() < m_minXPlanes ) continue;
            if ( hasT1T2T3Track( sciFiHits, itT1.hits() ) ) continue;
            PrTrackFitterXZ fitter;
            bool            fitX = fitter.fit( sciFiHits, itT1 );
            if ( fitX ) {
              auto worstHitX = findWorstX( itT1, nParts );
              if ( worstHitX == itT1.hits().end() ) { itT1.setRecovered( true ); }
            }
          } else {
            itT1.setRecovered( true );
          }
        }
      }
    }
    FTHitHandler = consolidateHitHandler( FTHitHandler );
    updateSmallDexes( smallDexes, FTHitHandler );
    for ( unsigned int part = 0; part < nParts; ++part ) {
      removeClonesRecover( part, tracksToRecover, xCandidates, m_removeClones_distance_recover );
      addStereo<RecoverCase> addStereoReco( *this );
      addStereoReco( sciFiHits, zoneCache, part, FTHitHandler, trackCandidates, xCandidates );
    }
  }

  void HybridSeeding::flagHits( unsigned int icase, unsigned int part, TrackCandidates& trackCandidates,
                                unsigned int firstTrackCand, const FT::Hits& sciFiHits, const SmallDexes& smallDexes,
                                PrFTHitHandler<ModPrHit>& hitHandler ) const noexcept {
    // The hits in the track candidates are copies, but they each contain their own index
    // in the original hit container, which is used to flag the original hits.
    for ( unsigned int i = firstTrackCand; i < trackCandidates[part].size(); i++ ) {
      auto& track = trackCandidates[part][i];
      if ( !track.valid() ) continue;
      if ( ( track.size() == 12 ) || ( ( track.size() == 11 && track.chi2PerDoF() < m_MaxChi2Flag[icase] &&
                                         std::fabs( track.ax() - track.bx() * Pr::Hybrid::zReference +
                                                    track.cx() * m_ConstC ) < m_MaxX0Flag[icase] ) ) ) {
        for ( const auto& hit : track.hits() ) {

          if ( const auto yMax = sciFiHits.coldHitInfo( hit.fullDex ).yMax;
               ( !part && yMax < 1000.f ) || ( part && yMax > 1000.f ) ) // DO NOT FLAG HITS FROM THE
                                                                         // OTHER PART FIXME
          {
            continue;
          }
          hitHandler.hit( smallDexes[hit.fullDex] ).setInvalid();
        }
      }
    }
  }

  //=========================================================================
  //  Convert to LHCb tracks
  //=========================================================================
  //---LoH: there are some things to understand, such as:
  // - does the scaleFactor or the momentumScale change? If not, momentumScale/(-scaleFactor) is much better.
  void HybridSeeding::makeLHCbTracks( const FT::Hits& sciFiHits, Tracks& result, unsigned int part,
                                      const TrackCandidates& trackCandidates, const DeMagnet& magnet ) const noexcept {
    unsigned int nTracks( 0 );
    const float val_z = 7826.1;
    for ( const auto& track : trackCandidates[part] ) {
      if ( !track.valid() ) continue;
      // AdditionalSeed Classifier
      const float nSciFiHits =  track.hits().size();
      const float tx = track.xSlopeFromDz( val_z - Pr::Hybrid::zReference );
      const float ty = track.ySlope();
      const float position_x = track.xFromDz( val_z - Pr::Hybrid::zReference );
      const float position_y = track.yFromDz( val_z - Pr::Hybrid::zReference );
      const float theta = std::atan(std::sqrt(tx * tx + ty * ty));
      const float eta = -std::log(std::tan(theta / 2));
      const float phi = std::atan2(ty, tx);
      const float chi2PerDoF = track.chi2PerDoF();


      // Min and max values for scaling.
      const std::vector<float> scaler_min = {9.00000000e+00, -1.04165041e+00, -3.68108153e-01, -2.64506885e+03,
        -2.06981274e+03,  8.29611024e-01, -3.14159233e+00,  1.61898648e-03};
      const std::vector<float> scaler_max = {1.20000000e+01, 1.06449950e+00, 3.93117666e-01, 2.63963184e+03,
        2.06795142e+03, 9.60124050e+00, 3.14159101e+00, 4.99820805e+00};

      // Scale each feature
      const std::vector<float> scaled_features = {
          minMaxScale(nSciFiHits, scaler_min[0], scaler_max[0]),
          minMaxScale(tx, scaler_min[1], scaler_max[1]),
          minMaxScale(ty, scaler_min[2], scaler_max[2]),
          minMaxScale(position_x, scaler_min[3], scaler_max[3]),
          minMaxScale(position_y, scaler_min[4], scaler_max[4]),
          minMaxScale(eta, scaler_min[5], scaler_max[5]),
          minMaxScale(phi, scaler_min[6], scaler_max[6]),
          minMaxScale(chi2PerDoF, scaler_min[7], scaler_max[7])
          // Add additional scaled features if necessary
      };

      // Pass the scaled values to the model
      const float rawValue = ApplySciFiModel(scaled_features);
      
      // Compute the probability using the sigmoid function
      const float probability = 1.0f / (1.0f + std::exp(-rawValue));

      // Apply the probability condition
      if (probability < 0.2) continue;
      auto         outTrack = result.emplace_back<SIMDWrapper::InstructionSet::Scalar>();
      unsigned int iID( 0 );
      auto         zFirstMeas = std::numeric_limits<float>::max();
      auto         zLastMeas  = std::numeric_limits<float>::lowest();
      outTrack.field<SeedTag::FTHits>().resize( track.hits().size() );
      for ( auto const& hit : track.hits() ) {
        const auto lhcbid =
            LHCb::Event::lhcbid_v<SIMDWrapper::scalar::types>( sciFiHits.lhcbid( hit.fullDex ).lhcbID() );
        outTrack.field<SeedTag::FTHits>()[iID].template field<SeedTag::LHCbID>().set( lhcbid );
        outTrack.field<SeedTag::FTHits>()[iID].template field<SeedTag::Index>().set( hit.fullDex );
        zFirstMeas = std::min( sciFiHits.z( hit.fullDex ), zFirstMeas );
        zLastMeas  = std::max( sciFiHits.z( hit.fullDex ), zLastMeas );
        iID++;
      }

      auto stateZref = State{
          { { track.ax(), track.yRef(), track.xSlope0(), track.ySlope(), std::numeric_limits<float>::signaling_NaN() },
            Pr::Hybrid::zReference } };
      double      qOverP, sigmaQOverP;
      const float scaleFactor = magnet.signedRelativeCurrent();
      if ( m_momentumTool->calculate( magnet, &stateZref, qOverP, sigmaQOverP, true ).isFailure() ) {
        if ( std::abs( scaleFactor ) < 1.e-4f ) {
          qOverP = ( ( track.cx() < 0.f ) ? -1.f : 1.f ) * ( ( scaleFactor < 0.f ) ? -1.f : 1.f ) / Gaudi::Units::GeV;
          sigmaQOverP = 1.f / Gaudi::Units::MeV;
        } else {
          qOverP      = track.cx() * m_momentumScale / ( -1.f * scaleFactor );
          sigmaQOverP = 0.5f * qOverP;
        }
      }

      constexpr auto state_locs =
          Event::v3::get_state_locations<Event::v3::available_states_t<Event::Enum::Track::Type::Ttrack>>{};
      for ( const auto loc : state_locs() ) {
        // TODO: C++23 or_else
        const auto z = [&] {
          if ( const auto z = Z( loc ); z.has_value() ) { return static_cast<float>( z.value() ); }
          switch ( loc ) {
          case Event::Enum::State::Location::FirstMeasurement:
            return zFirstMeas;
          case Event::Enum::State::Location::LastMeasurement:
            return zLastMeas;
          default:
            throw GaudiException( toString( loc ) + " is currently not created for PrSeedTracks", this->name(),
                                  StatusCode::FAILURE );
          }
        }();

        auto state = outTrack.field<SeedTag::States>( stateIndex<Event::Enum::Track::Type::Ttrack>( loc ) );
        state.setPosition( track.xFromDz( z - Pr::Hybrid::zReference ), track.yFromDz( z - Pr::Hybrid::zReference ),
                           z );
        state.setDirection( track.xSlopeFromDz( z - Pr::Hybrid::zReference ), track.ySlope() );
        state.setQOverP( static_cast<float>( qOverP ) );
      }
      outTrack.field<SeedTag::Chi2PerDoF>().set( track.chi2PerDoF() );
      nTracks++;
    }
    m_outputTracksCnt += nTracks;
  }

  //=================================================================================================

  //=========================================================================
  //  Finding the X projections. This is the most time-consuming part of the algorithm
  //=========================================================================

  void HybridSeeding::findXProjections( const FT::Hits& sciFiHits, const ZoneCache& zoneCache, unsigned int part,
                                        unsigned int iCase, const PrFTHitHandler<ModPrHit>& FTHitHandler,
                                        XCandidates& xCandidates ) const noexcept {
    // Counters
    auto twoHitCombCnt   = m_twoHitCombCnt_cases[iCase].buffer();
    auto threeHitCombCnt = m_threeHitCombCnt_cases[iCase].buffer();
    auto XZTrackCandCnt  = m_fullHitCombCnt_cases[iCase].buffer();

    auto&          xCands = xCandidates[part];
    CaseGeomInfoXZ xZones = initializeXProjections( zoneCache, iCase, part );
    float slope = ( m_tolAtX0Cut[iCase] - m_tolX0SameSign[iCase] ) / ( m_x0Cut[iCase] - m_x0SlopeChange[iCase] );
    float slopeopp =
        ( m_tolAtx0CutOppSign[iCase] - m_tolX0OppSign[iCase] ) / ( m_x0Cut[iCase] - m_x0SlopeChange2[iCase] );
    float accTerm1 = slope * m_x0SlopeChange[iCase] - m_tolX0SameSign[iCase];
    float accTerm2 = slopeopp * m_x0SlopeChange2[iCase] - m_tolX0OppSign[iCase];

    //======================================================================================================================================================
    SearchWindowsX searchWindows;
    SearchWindowsX tempWindows;
    //============sInitialization
    ZoneLimitsX zoneLimits;
    // Always: first=0 last=1 middles=2,3 remains=4,5 (order of loops)
    for ( unsigned int i = 0; i < LHCb::Detector::FT::nXLayersTotal; ++i ) {
      float lastHitX( -5000.f );
      auto  r = FTHitHandler.hits( xZones.zones[i] );
      if ( i == 0 )
        searchWindows[i] = { r.begin(), r.end(), std::numeric_limits<float>::lowest() };
      else
        searchWindows[i] = { r.begin(), r.begin(), std::numeric_limits<float>::lowest() };
      tempWindows[i] = { r.begin(), r.begin(), std::numeric_limits<float>::lowest() };
      // Store the coordinate of the last hit for faster methods
      if ( !r.empty() ) lastHitX = r.back().coord;
      zoneLimits[i] = { std::reverse_iterator{ r.begin() }, r.end(), lastHitX };
    }
    ModPrHitConstIter Fhit, Lhit;
    // Used to cache the position to do a "look-around search"
    std::array<SeedTrackParabolaHits, 2> parabolaSeedHits; // 0: layer T2x1, 1: layer T2x2, 2: extrapolated T2x2
    parabolaSeedHits[0].reserve( maxParabolaHits );
    parabolaSeedHits[1].reserve( maxParabolaHits );
    TwoHitCombination hitComb;
    float             tolHp = m_TolFirstLast[iCase];
    if ( m_minP > 0.f ) {
      float kDelta = xZones.zLays[0] * xZones.invZlZf; //---LoH: x0 = kDelta * DeltaInf
      float delta  = m_pFromTwoHitP1 * m_pFromTwoHitP1 +
                    ( 4.f * m_pFromTwoHitP2 / m_minP ); //--LoH: discriminant of the polynomial
      float tol2Hit = ( 1.f / kDelta ) * ( -m_pFromTwoHitP1 + std::sqrt( delta ) ) / 2.f / m_pFromTwoHitP2;
      tolHp         = ( tolHp < tol2Hit ) ? tolHp : tol2Hit;
    }
    for ( ; searchWindows[0].begin != searchWindows[0].end; ++searchWindows[0].begin ) // for a hit in first layer
    {
      Fhit            = searchWindows[0].begin;
      float xFirst    = Fhit->coord;
      float tx_inf    = xFirst * xZones.invZf;
      float xProjeInf = tx_inf * xZones.zLays[1];
      float maxXl     = xProjeInf + tx_inf * m_alphaCorrection[iCase] + tolHp;
      float minXl     = maxXl - 2.f * tolHp;
      // Setting the last-layer bounds
      searchWindows[1].begin =
          get_lowerBound_lin( searchWindows[1].begin, zoneLimits[1].end, minXl ); //---LoH: should be very small
      searchWindows[1].end =
          get_upperBound_lin( searchWindows[1].end, zoneLimits[1].end, maxXl ); //---LoH: between 6 and 20 times roughly
      // Reinitialise the parabola seed bounds to the highest previous point
      tempWindows[2].begin = searchWindows[2].begin;
      tempWindows[3].begin = searchWindows[3].begin;
      bool first           = true;
      twoHitCombCnt += ( searchWindows[1].end - searchWindows[1].begin );
      for ( Lhit = searchWindows[1].begin; Lhit != searchWindows[1].end; ++Lhit ) { // for a hit in last layer
        //---LoH: For 100 events, this is ran 1.6M times (number of updateXZCombination)
        float xLast = Lhit->coord;
        updateXZCombinationPars( iCase, xFirst, xLast, xZones, slope, slopeopp, accTerm1, accTerm2,
                                 hitComb ); // New parameters
        // Look for parabola hits and update bounds
        float xProjectedCorrected = xZones.zLays[2] * hitComb.tx + hitComb.x0new; // Target
        float xMin                = xProjectedCorrected + hitComb.minPara;
        float xMax                = xProjectedCorrected + hitComb.maxPara;
        bool  OK                  = findParabolaHits( xMin, xMax, zoneLimits[2], tempWindows[2], parabolaSeedHits[0] );
        float t2Dist              = hitComb.tx * xZones.t2Dist;
        OK += findParabolaHits( xMin + t2Dist, xMax + t2Dist, zoneLimits[3], tempWindows[3], parabolaSeedHits[1] );
        if ( first ) {
          searchWindows[2].begin = tempWindows[2].begin;
          searchWindows[3].begin = tempWindows[3].begin;
          first                  = false;
        }
        if ( !OK ) continue;
        // First parabola
        hitComb.xRef = hitComb.x0 + hitComb.tx * Pr::Hybrid::zReference;
        hitComb.x0 += xZones.zLays[2] * hitComb.tx; // x0 is now the projected x on T2x1
        unsigned int n0 = xCands.size();
        threeHitCombCnt += parabolaSeedHits[0].size();
        for ( unsigned int i = 0; i < parabolaSeedHits[0].size(); ++i ) {
          fillXhits0( sciFiHits, iCase, Fhit, parabolaSeedHits[0][i], Lhit, xZones, hitComb, zoneLimits, xCands,
                      tempWindows, XZTrackCandCnt );
        }
        //---LoH: do not even try to reconstruct a track if you already got one before.
        if ( n0 != xCands.size() ) { continue; }
        // First parabola
        hitComb.x0 += xZones.t2Dist * hitComb.tx; // x0 is now the projected x on T2x2
        for ( unsigned int i = 0; i < parabolaSeedHits[1].size(); ++i ) {
          fillXhits1( sciFiHits, iCase, Fhit, parabolaSeedHits[1][i], Lhit, xZones, hitComb, zoneLimits, xCands,
                      tempWindows, XZTrackCandCnt );
        }
      }
    } // end loop first zone
  }

  // Builds the parabola solver from solely z information
  //---LoH: Can be put outside of the class if m_dRatio is externalised

  // Fill hits due to parabola seeds in T2x1
  //---LoH: Can be put outside of the class.
  //---LoH: Can be templated
  template <typename CountingBuffer>
  void HybridSeeding::fillXhits0( const FT::Hits& sciFiHits, unsigned int iCase, ModPrHitConstIter Fhit,
                                  const ModPrHit& Phit, ModPrHitConstIter Lhit, const CaseGeomInfoXZ& xZones,
                                  const TwoHitCombination& hitComb, const ZoneLimitsX& zoneLimits,
                                  std::vector<Pr::Hybrid::SeedTrackX>& xCandidates, SearchWindowsX& searchWindows,
                                  CountingBuffer& XZTrackCandCnt ) const noexcept {
    //---LoH: called 712k times in 100 events
    SeedTrackHitsX xHits;
    float          tol = m_tolRemaining[iCase];
    float          c   = ( hitComb.x0 - Phit.coord ) * xZones.delSeedCorr1;
    float          b   = hitComb.tx - c * xZones.txCorr;
    float          a   = hitComb.xRef - c * xZones.xRefCorr;
    fillXhitParabola( tol, c * xZones.dz2Lays[3] + b * xZones.dzLays[3] + a, searchWindows[3], xHits );
    fillXhitRemaining( tol, c * xZones.dz2Lays[4] + b * xZones.dzLays[4] + a, searchWindows[4], zoneLimits[4], xHits );
    if ( xHits.size() == 0 ) return; //---LoH: ~650k times out of 750k
    // Add the last hit
    fillXhitRemaining( tol, c * xZones.dz2Lays[5] + b * xZones.dzLays[5] + a, searchWindows[5], zoneLimits[5], xHits );
    if ( xHits.size() < 2 ) return; //---LoH: ~50k times out of 750k
    xHits.push_back( *Fhit );
    xHits.push_back( Phit ); // T2x1
    xHits.push_back( *Lhit );
    XZTrackCandCnt += 1;
    Pr::Hybrid::SeedTrackX xCand( m_dRatio.value(), xHits );
    createXTrack( sciFiHits, iCase, xCandidates, xCand ); //---LoH: called 27k times
    return;
  }

  // Fill hits due to parabola seeds in T2x2
  //---LoH: Can be put outside of the class.
  //---LoH: Can be templated
  template <typename CountingBuffer>
  void HybridSeeding::fillXhits1( const FT::Hits& sciFiHits, unsigned int iCase, ModPrHitConstIter Fhit,
                                  const ModPrHit& Phit, ModPrHitConstIter Lhit, const CaseGeomInfoXZ& xZones,
                                  const TwoHitCombination& hitComb, const ZoneLimitsX& zoneLimits,
                                  std::vector<Pr::Hybrid::SeedTrackX>& xCandidates, SearchWindowsX& searchWindows,
                                  CountingBuffer& XZTrackCandCnt ) const noexcept {
    //---LoH: called 716k times
    SeedTrackHitsX xHits;
    float          tol = m_tolRemaining[iCase];
    float          c   = ( hitComb.x0 - Phit.coord ) * xZones.delSeedCorr2;
    float          b   = hitComb.tx - c * xZones.txCorr;
    float          a   = hitComb.xRef - c * xZones.xRefCorr;
    // a + b + c*(rfcorr+ txcorrz + z2)
    fillXhitRemaining( tol, c * xZones.dz2Lays[4] + b * xZones.dzLays[4] + a, searchWindows[4], zoneLimits[4], xHits );
    if ( xHits.size() != 1 ) return;
    fillXhitRemaining( tol, c * xZones.dz2Lays[5] + b * xZones.dzLays[5] + a, searchWindows[5], zoneLimits[5], xHits );
    if ( xHits.size() != 2 ) return;
    xHits.push_back( *Fhit );
    xHits.push_back( Phit ); // T2x2
    xHits.push_back( *Lhit );
    XZTrackCandCnt += 1;
    Pr::Hybrid::SeedTrackX xCand( m_dRatio.value(), xHits );
    createXTrack( sciFiHits, iCase, xCandidates, xCand );
  }

  void HybridSeeding::createXTrack( const FT::Hits& sciFiHits, unsigned int iCase,
                                    std::vector<Pr::Hybrid::SeedTrackX>& xCandidates,
                                    Pr::Hybrid::SeedTrackX&              xCand ) const noexcept {
    //---LoH: called 27k times
    //---LoH: out of all the calls, less than 50% refused
    // Create the track
    PrTrackFitterXZ fitter;
    bool            fitX = fitter.fit( sciFiHits, xCand );
    if ( !fitX ) { return; }
    // Algorithm allows to go down to 4 hits (m_minXPlanes) only from 6 hits ( == maximum ) track refit by construction.
    auto worstHitX     = findWorstX( xCand, iCase );
    bool canRemoveHitX = xCand.size() == LHCb::Detector::FT::nXLayersTotal;
    while ( worstHitX != xCand.hits().cend() ) {
      if ( !canRemoveHitX ) { return; }
      xCand.hits().erase( worstHitX );
      fitX = fitter.fit( sciFiHits, xCand );
      if ( !fitX ) { return; }
      worstHitX     = findWorstX( xCand, iCase );
      canRemoveHitX = xCand.size() > m_minXPlanes;
    }

    // Fit is successful and  make a x/z-candidate
    if ( ( ( xCand.chi2PerDoF() < m_maxChi2DoFX[iCase] ) ) ) {
      xCand.setXT1( xCand.xFromDz( StateParameters::ZBegT - Pr::Hybrid::zReference ) );
      xCand.setXT3( xCand.xFromDz( StateParameters::ZEndT - Pr::Hybrid::zReference ) );
      xCandidates.push_back( xCand );
    }
  }

  SeedTrackHitsConstIter HybridSeeding::findWorstX( const Pr::Hybrid::SeedTrackX& track, int iCase ) const noexcept {
    const auto worst_it  = std::max_element( track.chi2Hits().cbegin(), track.chi2Hits().cbegin() + track.size() );
    auto       worst_pos = worst_it - track.chi2Hits().cbegin();
    if ( track.chi2Hits()[worst_pos] < m_maxChi2HitsX[iCase] ) {
      return track.hits().cend();
    } else {
      return track.hits().cbegin() + worst_pos;
    }
  }

  void HybridSeeding::fillXhitParabola( float tol, float xAtZ, const SearchWindow& searchWindow,
                                        SeedTrackHitsX& xHits ) const noexcept {
    //---LoH: called 712k times in 100 events
    // Prediction of the position with the cubic correction!
    float xMinAtZ = xAtZ - tol; // * (1. + dz*invZref );
    // Parabola here is larger than tolerances in x0! ( 0.75 mm up / down when x0 ~0 )
    // may we want ot make the m_tolRemaining[iCase] ( x0 dependent too ?? )
    // here you are re-using the tolerance defined before from the linear extrapolation of 2-hit combo
    ModPrHitConstIter bestProj = searchWindow.end;
    float             tmpDist;
    for ( auto itH = searchWindow.begin; itH != searchWindow.end; ++itH ) {
      if ( itH->coord < xMinAtZ ) { continue; } // this condition relies on the fact that invalid hits are set to -inf
      tmpDist = std::fabs( itH->coord - xAtZ );
      if ( tmpDist < tol ) {
        tol      = tmpDist;
        bestProj = itH;
      } else
        break;
    }
    // Can happen if there is only one hit in the loop.
    if ( bestProj != searchWindow.end ) {
      xHits.push_back( *bestProj );
      return;
    }
    return;
  }

  void HybridSeeding::fillXhitRemaining( float tol, float xAtZ, SearchWindow& searchWindow, const ZoneLimits& zoneLimit,
                                         SeedTrackHitsX& xHits ) const noexcept {
    //---LoH: called 1.66M times
    // Prediction of the position with the cubic correction!
    float xMinAtZ = xAtZ - tol;
    LookAroundMin( searchWindow, xMinAtZ, zoneLimit );
    ModPrHitConstIter bestProj = zoneLimit.end;
    float             tmpDist;
    for ( searchWindow.end = searchWindow.begin; searchWindow.end < zoneLimit.end; ++searchWindow.end ) {
      tmpDist = std::fabs( searchWindow.end->coord - xAtZ );
      if ( tmpDist < tol ) {
        tol      = tmpDist;
        bestProj = searchWindow.end;
      } else
        break;
    }
    if ( bestProj != zoneLimit.end ) { xHits.push_back( *bestProj ); } // todo
    return;
  }

  void HybridSeeding::updateXZCombinationPars( unsigned int iCase, float xFirst, float xLast,
                                               const CaseGeomInfoXZ& xZones, float slope, float slopeopp,
                                               float accTerm1, float accTerm2,
                                               TwoHitCombination& hitComb ) const noexcept {
    //---LoH: Called 3M times in 100 events
    hitComb.tx    = ( xLast - xFirst ) * xZones.invZlZf;   //---LoH: this is basically b
    hitComb.x0    = xFirst - hitComb.tx * xZones.zLays[0]; //---LoH: this carries all the momentum information
    hitComb.x0new = hitComb.x0 * ( m_x0Corr[iCase] );
    if ( hitComb.x0 > 0.f ) {
      hitComb.minPara = hitComb.x0 > m_x0SlopeChange[iCase] ? -slope * hitComb.x0 + accTerm1 : -m_tolX0SameSign[iCase];
      hitComb.maxPara =
          hitComb.x0 > m_x0SlopeChange2[iCase] ? slopeopp * hitComb.x0 - accTerm2 : +m_tolX0OppSign[iCase];
    } else {
      hitComb.maxPara = hitComb.x0 < -m_x0SlopeChange[iCase] ? -slope * hitComb.x0 - accTerm1 : m_tolX0SameSign[iCase];
      hitComb.minPara =
          hitComb.x0 < -m_x0SlopeChange2[iCase] ? slopeopp * hitComb.x0 + accTerm2 : -m_tolX0OppSign[iCase];
    }
  }

  CaseGeomInfoXZ HybridSeeding::initializeXProjections( ZoneCache const& zoneCache, unsigned int iCase,
                                                        unsigned int part ) const noexcept {
    CaseGeomInfoXZ xZones;
    unsigned int   firstZoneId( 0 ), lastZoneId( 0 );
    if ( 0 == iCase ) {
      firstZoneId = LHCb::Detector::FT::UpperZones::T1X1 - part;
      lastZoneId  = LHCb::Detector::FT::UpperZones::T3X2 - part;
    } else if ( 1 == iCase ) {
      firstZoneId = LHCb::Detector::FT::UpperZones::T1X2 - part;
      lastZoneId  = LHCb::Detector::FT::UpperZones::T3X1 - part;
    } else if ( 2 == iCase ) {
      firstZoneId = LHCb::Detector::FT::UpperZones::T1X1 - part;
      lastZoneId  = LHCb::Detector::FT::UpperZones::T3X1 - part;
    } else if ( 3 == iCase ) {
      firstZoneId = LHCb::Detector::FT::UpperZones::T1X2 - part;
      lastZoneId  = LHCb::Detector::FT::UpperZones::T3X2 - part;
    }

    // Array[0] = first layer in T1 for 2-hit combo
    xZones.zones[0]   = firstZoneId;
    xZones.zLays[0]   = zoneCache.z( firstZoneId / 2 );
    xZones.dzLays[0]  = xZones.zLays[0] - Pr::Hybrid::zReference;
    xZones.dz2Lays[0] = xZones.dzLays[0] * xZones.dzLays[0] * ( 1.f + m_dRatio * xZones.dzLays[0] );
    // Array[1] = last  layer in T3 for 2-hit combo
    xZones.zones[1]   = lastZoneId;
    xZones.zLays[1]   = zoneCache.z( lastZoneId / 2 );
    xZones.dzLays[1]  = xZones.zLays[1] - Pr::Hybrid::zReference;
    xZones.dz2Lays[1] = xZones.dzLays[1] * xZones.dzLays[1] * ( 1.f + m_dRatio * xZones.dzLays[1] );

    // Array[2] = T2-1st x-layers
    xZones.zones[2]   = LHCb::Detector::FT::UpperZones::T2X1 - part;
    xZones.zLays[2]   = zoneCache.z( LHCb::Detector::FT::UpperZones::T2X1 / 2 );
    xZones.dzLays[2]  = xZones.zLays[2] - Pr::Hybrid::zReference;
    xZones.dz2Lays[2] = xZones.dzLays[2] * xZones.dzLays[2] * ( 1.f + m_dRatio * xZones.dzLays[2] );
    // Array[3] = T2-2nd x-layers
    xZones.zones[3]   = LHCb::Detector::FT::UpperZones::T2X2 - part;
    xZones.zLays[3]   = zoneCache.z( LHCb::Detector::FT::UpperZones::T2X2 / 2 );
    xZones.dzLays[3]  = xZones.zLays[3] - Pr::Hybrid::zReference;
    xZones.dz2Lays[3] = xZones.dzLays[3] * xZones.dzLays[3] * ( 1.f + m_dRatio * xZones.dzLays[3] );

    unsigned int i = 4;
    // Add extra layer HERE, if needed!!!!
    for ( unsigned int xZoneId : { LHCb::Detector::FT::UpperZones::T1X1, LHCb::Detector::FT::UpperZones::T1X2,
                                   LHCb::Detector::FT::UpperZones::T3X1, LHCb::Detector::FT::UpperZones::T3X2 } ) {
      xZoneId = xZoneId - part;
      if ( xZoneId != firstZoneId && xZoneId != lastZoneId ) {
        xZones.zones[i]   = xZoneId;
        xZones.zLays[i]   = zoneCache.z( xZoneId / 2 );
        xZones.dzLays[i]  = xZones.zLays[i] - Pr::Hybrid::zReference;
        xZones.dz2Lays[i] = xZones.dzLays[i] * xZones.dzLays[i] * ( 1.f + m_dRatio * xZones.dzLays[i] );
        ++i;
      }
    }
    xZones.invZf   = 1.f / xZones.zLays[0];
    xZones.invZlZf = 1.f / ( xZones.zLays[1] - xZones.zLays[0] );

    // Parabola z difference
    xZones.t2Dist = xZones.zLays[3] - xZones.zLays[2];

    // Correction factors
    // xInfCorr  = (zLast/zFirst) -1
    // x0        = Del_x_inf*xInfCorr
    //  xZones.xInfCorr = -1.f/((xZones.dzLays[1]/xZones.dzLays[0]) - 1.f);
    // tx_picked = b + c * txCorr (NEEDS TO BE CORRECTED AS WELL WITH D)
    //  xZones.txCorr         = (xZones.dz2Lays[1]-xZones.dz2Lays[0])/(xZones.zLays[1]-xZones.zLays[0]);
    xZones.txCorr = xZones.dzLays[1] + xZones.dzLays[0] +
                    m_dRatio * ( xZones.dzLays[1] * xZones.dzLays[1] + xZones.dzLays[0] * xZones.dzLays[0] +
                                 xZones.dzLays[1] * xZones.dzLays[0] );
    // xRef      = a + c * xRefCorr
    xZones.xRefCorr =
        xZones.dzLays[0] * ( xZones.dzLays[0] - xZones.txCorr + m_dRatio * xZones.dzLays[0] * xZones.dzLays[0] );

    // DeltaSeed_X = c*delSeedCorrX
    xZones.delSeedCorr1 =
        1.f /
        ( ( ( ( xZones.zLays[2] - xZones.zLays[0] ) * xZones.invZlZf ) * ( xZones.dz2Lays[1] - xZones.dz2Lays[0] ) ) +
          ( xZones.dz2Lays[0] - xZones.dz2Lays[2] ) );
    xZones.delSeedCorr2 =
        1.f /
        ( ( ( ( xZones.zLays[3] - xZones.zLays[0] ) * xZones.invZlZf ) * ( xZones.dz2Lays[1] - xZones.dz2Lays[0] ) ) +
          ( xZones.dz2Lays[0] - xZones.dz2Lays[3] ) );

    return xZones;
  }
} // namespace LHCb::Pr
