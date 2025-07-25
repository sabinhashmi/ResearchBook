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

// standard
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

// boost
#include "boost/container/static_vector.hpp"

// vdt
#include "vdt/log.h"

// from Gaudi
#include "GaudiAlg/ISequencerTimerTool.h"
#include "GaudiKernel/ToolHandle.h"

// from LHCb
#include "Core/FloatComparison.h"
#include "DetDesc/GenericConditionAccessorHolder.h"
#include "Event/PrLongTracks.h"
#include "Event/PrSciFiHits.h"
#include "Event/StateParameters.h"
#include "Event/StateVector.h"
#include "Event/Track.h"
#include "FTDet/DeFTDetector.h"
#include "Kernel/CountIterator.h"
#include "LHCbAlgs/Transformer.h"
#include "LHCbMath/SIMDWrapper.h"
#include "Magnet/DeMagnet.h"

// local
#include "HoughSearch.h"
#include "PrKernel/FTGeometryCache.h"
#include "PrKernel/IPrAddUTHitsTool.h"
#include "PrKernel/IPrDebugTrackingTool.h"
#include "PrTrackModel.h"

// NN for ghostprob
#include "weights/TMVA_MLP_GhostNN_PrForwardTracking.h"
#include "weights/TMVA_MLP_GhostNN_PrForwardTrackingVelo.h"

/**
  ++++++++++++++++++++++++++++++++ Forward Tracking +++++++++++++++++++++++++++++++
*/

/** +++++++++++++++++++++ Short Introduction in geometry ++++++++++++++++++++++++++:
 *
 * The SciFi Tracker Detector, or simple Fibre Tracker (FT) consits out of 3 stations.
 * Each station consists out of 4 planes/layers. Thus there are in total 12 layers,
 * in which a particle can leave a hit. The reasonable maximum number of hits a track
 * can have is thus also 12.
 *
 * Each layer consists out of several Fibre mats. A fibre has a diameter of below a mm.
 * Several fibres are glued alongside each other to form a mat.
 * A Scintilating Fibre produces light, if a charged particle traverses. This light is then
 * detected on the outside of the Fibre mat.
 *
 * Looking from the collision point, one (X-)layer looks like the following:
 *
 *                    y       6m
 *                    ^  ||||||||||||| Upper side
 *                    |  ||||||||||||| 2.5m
 *                    |  |||||||||||||
 *                   -|--||||||o||||||----> -x
 *                       |||||||||||||
 *                       ||||||||||||| Lower side
 *                       ||||||||||||| 2.5m
 *
 * All fibres are aranged parallel to the y-axis. There are three different
 * kinds of layers, denoted by X,U,V. The U/V layers are rotated with respect to
 * the X-layers by +/- 5 degrees, to also get a handle of the y position of the
 * particle. As due to the magnetic field particles are deflected in
 * x-direction (only small deflection in y-direction).
 * The layer structure in the FT is XUVX-XUVX-XUVX.
 *
 * The detector is divided into an upeer and a lower side (>/< y=0). As particles
 * are deflected in x direction there are only very(!) few particles that go
 * from the lower to the upper side, or vice versa. The reconstruction algorithm therefore
 * treats upper and lower side tracks independently such that only hits in the respective
 * detector half are considered.
 * Only for U/V layers this can be different because in these layers the
 * complete fibre modules are rotated, producing a zic-zac pattern at y=0, also
 * called  "the triangles". In these layers hits are also searched for on the "other side",
 * if the track is close to
 * y=0. Sketch (rotation exagerated):
 *                                          _.*
 *     y ^   _.*                         _.*
 *       | .*._      Upper side       _.*._
 *       |     *._                 _.*     *._
 *       |--------*._           _.*           *._----------------> x
 *       |           *._     _.*                 *._     _.*
 *                      *._.*       Lower side      *._.*
 *
 *
 *
 *
 *
 *       Zone ordering defined in Detector/FT/FTConstants.h
 *
 *     y ^
 *       |    1  3  5  7     9 11 13 15    17 19 21 23
 *       |    |  |  |  |     |  |  |  |     |  |  |  |
 *       |    x  u  v  x     x  u  v  x     x  u  v  x   <-- type of layer
 *       |    |  |  |  |     |  |  |  |     |  |  |  |
 *       |------------------------------------------------> z
 *       |    |  |  |  |     |  |  |  |     |  |  |  |
 *       |    |  |  |  |     |  |  |  |     |  |  |  |
 *       |    0  2  4  6     8 10 12 14    16 18 20 22
 *
 *
 *
 *  A detailed introduction to the basic ideas of the Forward tracking can be
 *   found here:
 *   (2002) http://cds.cern.ch/record/684710/files/lhcb-2002-008.pdf
 *   (2007) http://cds.cern.ch/record/1033584/files/lhcb-2007-015.pdf
 *  The most recent note contains information about parameterisations
 *   (2023) https://cds.cern.ch/record/2865000
 *
 *
 * Method overview
 *
 * The track reconstruction is done in several steps:
 *
 * 1) Hough-like transformation of SciFi hits using input tracks
 *    -> projectXHitsToHoughSpace and projectStereoHitsToHoughSpace
 *       filling a histogram-like data structure
 * 2) Threshold scan over "histogram" selecting local accumulations of hits
 *    -> pickUpCandidateBins and sortAndCopyBinContents
 * 3) Detailed selection of sets of x hits matching the input track
 *    -> selextXCandidates
 * 4) Completion of candidates by finding best matching u/v hits
 *    -> selectFullCandidates
 *        also assigning a "quality" to each candidate indicating a ghost
 *        probability
 * 5) Removing duplicates among all found tracks
 *    -> removeDuplicates
 */

/** @class PrForwardTracking PrForwardTracking.cpp
 *
 *  @author Olivier Callot
 *  @date   2012-03-20
 *  @author Thomas Nikodem
 *  @date   2013-03-15
 *  @author Michel De Cian
 *  @date   2014-03-12 Changes with make code more standard and faster
 *  @author Sevda Esen
 *  @date   2015-02-13 additional search in the triangles by Marian Stahl
 *  @author Thomas Nikodem
 *  @date   2016-03-09 complete restructuring
 *  @author Olli Lupton
 *  @date   2018-11-07 Imported PrForwardTool into PrForwardTracking as a step towards making PrForwardTracking accept
 *                     selections
 *  @author André Günther
 *  @date 2019-12-03 adapted PrForwardTracking to new SoA hit class PrSciFiHits
 *  @date 2021-03-07 complete redesign
 */
namespace LHCb::Pr::Forward {

  namespace Hough {
    constexpr int DEPTH   = 2;
    constexpr int MAXCAND = 2;
    constexpr int NBINS   = 15;
  } // namespace Hough
  namespace {
    using namespace FT;
    using TracksTag       = Long::Tag;
    using TracksFT        = Long::Tracks;
    using TracksUT        = Upstream::Tracks;
    using TracksVP        = Velo::Tracks;
    using scalar          = SIMDWrapper::scalar::types;
    using simd            = SIMDWrapper::best::types;
    using PrForwardTracks = std::vector<PrForwardTrack>;
    using StereoSearch =
        ::Hough::HoughSearch<int, Hough::DEPTH, Hough::MAXCAND, Detector::FT::nUVLayersTotal, Hough::NBINS>;

    // names of variables used in multilayer perceptron used to identify ghosts
    constexpr std::array<std::string_view, 9> NNVars{ "redChi2",
                                                      "abs((x+(zMagMatch-770.0)*tx)-(xEndT+(zMagMatch-9410.0)*txEndT))",
                                                      "abs(ySeedMatch-yEndT)",
                                                      "abs(yParam0Final-yParam0Init)",
                                                      "abs(yParam1Final-yParam1Init)",
                                                      "abs(ty)",
                                                      "abs(qop)",
                                                      "abs(tx)",
                                                      "abs(xParam1Final-xParam1Init)" };

    constexpr std::array<std::string_view, 10> NNVarsVeloUT{
        "log(abs(1./qop-1./qopUT))",
        "redChi2",
        "abs((x+(zMagMatch-770.0)*tx)-(xEndT+(zMagMatch-9410.0)*txEndT))",
        "abs(ySeedMatch-yEndT)",
        "abs(yParam0Final-yParam0Init)",
        "abs(yParam1Final-yParam1Init)",
        "abs(ty)",
        "abs(qop)",
        "abs(tx)",
        "abs(xParam1Final-xParam1Init)" };

    /**
     * @return larger n such that it is divisible by simd::size and padded by one more simd::size
     */
    constexpr size_t alignAndPad( size_t n ) { return ( n / simd::size + 2 ) * simd::size; }

    /**
     * @class PrParametersX PrForwardTracking.cpp
     * @brief Bundles parameters used selection of x candidates
     */
    struct PrParametersX {
      unsigned minXHits{};
      float    maxXWindow{};
      float    maxXWindowSlope{};
      float    maxXGap{};
      unsigned minStereoHits{};
      float    maxChi2PerDoF{};
      float    maxChi2XProjection{};
      float    maxChi2LinearFit{};
    };
    /**
     * @class PrParametersY PrForwardTracking.cpp
     * @brief Bundles parameters used selection of stereo and full candidates
     */
    struct PrParametersY {
      float    maxTolY{};
      float    uvSearchBinWidth{};
      float    tolY{};
      float    tolYSlope{};
      float    yTolUVSearch{};
      float    tolYTriangleSearch{};
      unsigned minStereoHits{};
      float    maxChi2StereoLinear{};
      float    maxChi2Stereo{};
      float    maxChi2StereoAdd{};
      float    tolYMag{};
      float    tolYMagSlope{};
    };

    /**
     * @brief remove hits from range if found on track
     * @param idx1 first index of range
     * @param idxEnd end index of range
     * @param track track containing hits that are removed from range
     * @details Note that idxEnd is modified by this function because hits are removed by shuffling them to the
     * end of the range and simply moving the end index such that they are not contained anymore.
     * @return true if range is not empty after removing hits
     */
    auto removeUsedHits( int idx1, int& idxEnd, const XCandidate& track, ModSciFiHits::ModPrSciFiHitsSOA& allXHits ) {
      const auto& hitsOnTrack = track.getCoordsToFit();
      const auto  hits_simd   = allXHits.simd();
      for ( auto idx{ idx1 }; idx < idxEnd; idx += simd::size ) {
        const auto loopMask = simd::loop_mask( idx, idxEnd );
        auto       keepMask = loopMask;

        const auto hit_proxy = hits_simd[idx];
        const auto fulldexes = hit_proxy.get<ModSciFiHits::HitTag::fulldex>();
        const auto coords    = hit_proxy.get<ModSciFiHits::HitTag::coord>();

        for ( auto iHit : hitsOnTrack ) {
          const auto foundMask = fulldexes == iHit;
          keepMask             = keepMask && !foundMask;
        }

        allXHits.store<ModSciFiHits::HitTag::fulldex>( idx1, fulldexes, keepMask );
        allXHits.store<ModSciFiHits::HitTag::coord>( idx1, coords, keepMask );
        idx1 += popcount( keepMask );

        if ( any( !loopMask ) ) {
          const auto coords_new    = select( loopMask, hit_proxy.get<ModSciFiHits::HitTag::coord>(), coords );
          const auto fulldexes_new = select( loopMask, hit_proxy.get<ModSciFiHits::HitTag::fulldex>(), fulldexes );
          allXHits.store<ModSciFiHits::HitTag::fulldex>( idx, fulldexes_new );
          allXHits.store<ModSciFiHits::HitTag::coord>( idx, coords_new );
        }
      }
      const auto ok = idx1 != idxEnd;
      idxEnd        = idx1;
      return ok;
    }

    /**
     * @brief initialises parameters of x projection parameterisation
     */
    void initXFitParameters( XCandidate& trackCandidate, const VeloSeedExtended& veloSeed ) {
      const auto xAtRef           = trackCandidate.xAtRef();
      auto       dSlope           = ( xAtRef - veloSeed.xStraightAtRef ) / ( zReference - veloSeed.zMag );
      const auto zMag             = veloSeed.calcZMag( dSlope );
      const auto xMag             = veloSeed.seed.x( zMag );
      const auto slopeT           = ( xAtRef - xMag ) / ( zReference - zMag );
      dSlope                      = slopeT - veloSeed.seed.tx;
      const auto CX               = veloSeed.calcCX( dSlope );
      const auto DX               = veloSeed.calcDX( dSlope );
      trackCandidate.getXParams() = { { xAtRef, slopeT, CX, DX } };
    }

    /**
     * @brief initialises parameters of y projection parameterisation
     */
    void initYFitParameters( XCandidate& trackCandidate, const VeloSeedExtended& veloSeed ) {
      const auto dSlope           = trackCandidate.getXParams()[1] - veloSeed.seed.tx;
      const auto AY               = veloSeed.yStraightAtRef + veloSeed.calcYCorr( dSlope );
      const auto BY               = veloSeed.seed.ty + veloSeed.calcTyCorr( dSlope );
      const auto CY               = veloSeed.calcCY( dSlope );
      trackCandidate.getYParams() = { { AY, BY, CY } };
    }

    /**
     * @class HoughTransformation PrForwardTracking.cpp
     * @brief Contains data and methods used by the Hough-transformation-like step in the Forward Tracking
     * Main methods are defined outside of class
     * Short methods are defined inline
     */
    class HoughTransformation {
    public:
      static constexpr int      minBinOffset{ 2 };
      static constexpr int      nBins{ 1150 + minBinOffset };
      static constexpr int      reservedBinContent{ 16 };
      static constexpr float    rangeMax{ 3000.f * Gaudi::Units::mm };
      static constexpr float    rangeMin{ -3000.f * Gaudi::Units::mm };
      static constexpr unsigned nDiffPlanesBits{ 8 };
      static constexpr unsigned diffPlanesFlags{ 0xFF }; // 1111 1111
      static constexpr unsigned uvFlags{ 0x666 };        // 0110 0110 0110
      static constexpr unsigned xFlags{ 0x999 };         // 1001 1001 1001
      static_assert( sizeof( unsigned ) >= 4 );          // at the moment at least 32 bit datatype for plane encoding
      static_assert( ( reservedBinContent % simd::size == 0 ) &&
                     "Currently only multiples of avx2 vector width supported" );

      void projectStereoHitsToHoughSpace( const VeloSeedExtended&, const ZoneCache&, std::pair<float, float>,
                                          const FT::Hits& );
      void projectXHitsToHoughSpace( const VeloSeedExtended&, std::pair<float, float>, const FT::Hits& );
      auto pickUpCandidateBins( int, int, int, int );
      void sortAndCopyBinContents( int, ModSciFiHits::ModPrSciFiHitsSOA& );
      /**
       * @brief clears containers for size and therefore also content, plane counter and candidates
       */
      void clear() {
        m_candSize = 0;
        m_planeCounter.fill( 0 );
        m_binContentSize.fill( 0 );
      }

      /**
       * @brief Calculates bin numbers for x values in "Hough space".
       * @param xAtRef x values in "Hough space", i.e. on reference plane.
       * @param idx index of x hit
       * @param maxIdx last index/end of range
       * @return bin numbers
       * @details The binning is non-simple and follows the occupancy in the detector. Furthermore, only values within
       * a certain range are considered. x values outside of this range will cause a return of bin number 0, the
       * "garbage bin". The binning function is a fast sigmoid. The parameterisation is obtained using the
       * Reco-Parametersiation-Tuner.
       * https://gitlab.cern.ch/gunther/prforwardtracking-parametrisation-tuner/-/tree/master
       */
      auto calculateBinNumber( simd::float_v xAtRef, int idx, int maxIdx ) const {
        const auto boundaryMask = xAtRef < rangeMax && xAtRef > rangeMin && simd::loop_mask( idx, maxIdx );
        // p[0] + p[1] * x / (1 + p[2] * abs(x)) for nBins = 1152
        constexpr auto p = std::array{ 576.9713937732083f, 0.5780266207743059f, 0.0006728484590464921f };
        const auto     floatingBinNumber = p[0] + p[1] * xAtRef / ( 1.f + p[2] * abs( xAtRef ) );
        // it can happen that two almost equal xAtRef lead to floating bin numbers right on a bin edge, where one
        // falls into the left, the other into the right bin due to numerical differences coming from the calculations
        // above. This messes with the sorting that is only done per bin later on, but has no dramatic consequences
        // (the hits are so close to each other in that case that no deep logic breaks)
        const auto binNumber = static_cast<simd::int_v>( floatingBinNumber );
        return select( boundaryMask, binNumber, 0 );
      }

      /**
       * @brief Performs a compressed store of candidates and handles size incrementation.
       * @param mask Mask of candidates to store.
       * @param bins The bin number(s) qualifying as candidates.
       */
      void compressstoreCandidates( simd::mask_v mask, simd::int_v bins ) {
        bins.compressstore( mask, m_candidateBins.data() + m_candSize );
        m_candSize += popcount( mask );
      }

      /**
       * @brief Decodes number of different planes encoded in number stored in plane counting array.
       * @param bin The bin number
       */
      template <typename I>
      auto decodeNbDifferentPlanes( int bin ) const {
        return I{ m_planeCounter.data() + bin } & diffPlanesFlags;
      }

      /**
       * @brief return integer with bits encoding hit planes shifted to the beginning
       * @param bin The bin number
       */
      template <typename I>
      auto planeBits( int bin ) const {
        return I{ m_planeCounter.data() + bin } >> nDiffPlanesBits;
      }

      /**
       * @brief Make a span for one bin
       * @param iBin bin number
       * @return span for the coordinates in this bin
       */
      auto getBinContentCoord( int iBin ) {
        return LHCb::span{ m_binContentCoord }.subspan( iBin * reservedBinContent, m_binContentSize[iBin] );
      }
      auto getBinContentCoord( int iBin ) const {
        return LHCb::span{ m_binContentCoord }.subspan( iBin * reservedBinContent, m_binContentSize[iBin] );
      }

      /**
       * @brief Make a span for one bin
       * @param iBin bin number
       * @return span for the PrSciFiHits indices in this bin
       */
      auto getBinContentFulldex( int iBin ) {
        return LHCb::span{ m_binContentFulldex }.subspan( iBin * reservedBinContent, m_binContentSize[iBin] );
      }
      auto getBinContentFulldex( int iBin ) const {
        return LHCb::span{ m_binContentFulldex }.subspan( iBin * reservedBinContent, m_binContentSize[iBin] );
      }

      /**
       * @brief Sorts and then copies content of a bin to a new container
       * @param bin bin number
       * @param allXHits container to copy to (SoA)
       * @details The sorting is done by an insertion sort because the number of elements to sort is small by design.
       * The sorting is done by storing the permutations and then using a gather operation to shuffle according to the
       * permutations, followed by the storing.
       */
      [[gnu::always_inline]] void sortAndCopyBin( int bin, ModSciFiHits::ModPrSciFiHitsSOA& allXHits ) {
        if ( const auto binSize = m_binContentSize[bin]; binSize ) {
          std::iota( m_binPermutations.begin(), m_binPermutations.end(), 0 );
          const auto coordContent = getBinContentCoord( bin );
          for ( auto iCoord{ 1 }; iCoord < binSize; ++iCoord ) {
            const auto insertVal = coordContent[iCoord];
            auto       insertPos = iCoord;
            for ( auto movePos = iCoord; movePos && insertVal < coordContent[m_binPermutations[--movePos]];
                  --insertPos ) {
              m_binPermutations[insertPos] = m_binPermutations[movePos];
            }
            m_binPermutations[insertPos] = iCoord;
          }
          const auto fulldexContent = getBinContentFulldex( bin );
          const auto max_size       = allXHits.size() + binSize;
          auto       i{ 0 };
          do {
            const auto permut_v  = simd::int_v{ m_binPermutations.data() + i };
            const auto fulldex_v = gather( fulldexContent.data(), permut_v );
            const auto coord_v   = gather( coordContent.data(), permut_v );
            auto       hit       = allXHits.template emplace_back<>( max_size );
            hit.template field<ModSciFiHits::HitTag::fulldex>().set( fulldex_v );
            hit.template field<ModSciFiHits::HitTag::coord>().set( coord_v );
            i += simd::size;
          } while ( i < binSize );
        }
      }

      // for debugging
      [[maybe_unused]] friend std::ostream& operator<<( std::ostream& os, const HoughTransformation& h ) {
        const auto candidateBins = LHCb::span( h.m_candidateBins.begin(), h.m_candSize );
        os << "Found " << h.m_candSize << " candidate bins:" << std::endl;
        for ( auto bin : candidateBins ) {
          os << "Bin " << bin << " x coords: [ ";
          const auto coords    = h.getBinContentCoord( bin );
          const auto fulldexes = h.getBinContentFulldex( bin );
          for ( auto i{ 0 }; i < h.m_binContentSize[bin]; ++i ) {
            os << "(" << fulldexes[i] << ", " << coords[i] << ")";
            os << ( i != h.m_binContentSize[bin] - 1 ? " | " : "" );
          }
          os << "]"
             << " size=" << h.m_binContentSize[bin] << std::endl;
          const auto planes =
              ( h.m_planeCounter[bin] | h.m_planeCounter[bin - 1] | h.m_planeCounter[bin + 1] ) >> h.nDiffPlanesBits;
          os << "Used planes including neighbours: " << std::bitset<12>( planes ) << std::endl;
          os << std::endl;
        }
        return os;
      }

    private:
      /**
       * @property Contain the coordinates (x hits on reference plane) and PrSciFiHits indices in each bin
       */
      alignas( 64 ) std::array<float, alignAndPad( nBins ) * reservedBinContent> m_binContentCoord;
      alignas( 64 ) std::array<int, alignAndPad( nBins ) * reservedBinContent> m_binContentFulldex;
      /**
       * @property Contains the the number of unique planes within a bin encoded together with bit flags for each plane.
       */
      alignas( 64 ) std::array<int, alignAndPad( nBins )> m_planeCounter{};
      /**
       * @property Contains the number of hits in each bin
       */
      alignas( 64 ) std::array<int, alignAndPad( nBins )> m_binContentSize{};
      /**
       * @property Contains a selection of bins that qualify as part of a track candidates.
       * @details The number of candidates is tracked by candSize.
       */
      alignas( 64 ) std::array<int, alignAndPad( nBins )> m_candidateBins;
      /**
       * @property sstorage that contains temporary permutations from insertion sort
       */
      alignas( 64 ) std::array<int, reservedBinContent> m_binPermutations;
      int m_candSize{ 0 };
    };

    /**
     * @brief separate hits that are alone on their SciFi plane from ones that have friends on the plane
     * @param otherPlanes Container keeping track of the plane numbers containing multiples
     * @param protoCand the candidate under consideration
     * @details single hits are directly put into coordsToFit container inside protoCand
     */
    template <typename Container>
    [[gnu::always_inline]] inline void separateSingleHitsForFit( Container& otherPlanes, XCandidate& protoCand,
                                                                 const VeloSeedExtended& veloSeed ) {
      otherPlanes.clear();
      for ( unsigned int iPlane{ 0 }; iPlane < Detector::FT::nXLayersTotal; ++iPlane ) {
        if ( protoCand.planeSize( iPlane ) == 1 ) {
          const auto idx = protoCand.getIdx( iPlane * protoCand.planeMulti );
          protoCand.addHitForLineFit( idx, veloSeed );
        } else if ( protoCand.planeSize( iPlane ) ) {
          otherPlanes.push_back( iPlane );
        }
      }
    }

    /**
     * @brief add hits to candidate from planes that have multiple hits
     * @param otherPlanes Container keeping track of the plane numbers containing multiples
     * @param protoCand the candidate under consideration
     * @details only one hit per plane is allowed, selected by smallest chi2 calculated
     * using a straight line fit and putting them into coordsToFit container inside protoCand
     */
    template <typename Container>
    [[gnu::always_inline]] inline void addBestOtherHits( const Container& otherPlanes, XCandidate& protoCand,
                                                         const VeloSeedExtended& veloSeed ) {
      for ( auto iPlane : otherPlanes ) {
        const auto idxSpan = protoCand.getIdxSpan( iPlane );
        assert( idxSpan.size() > 0 );
        const auto bestIdx = [&] {
          auto best{ 0 };
          auto bestChi2{ std::numeric_limits<float>::max() };
          for ( auto idx : idxSpan ) {
            const auto chi2 = protoCand.lineChi2( idx, veloSeed );
            if ( chi2 < bestChi2 ) {
              bestChi2 = chi2;
              best     = idx;
            }
          }
          return best;
        }();
        protoCand.addHitForLineFit( bestIdx, veloSeed );
        protoCand.setPlaneSize( iPlane, 1 );
        protoCand.solveLineFit();
      }
    }

    /**
     * @brief store all hits present in candidate for fitting
     */
    void prepareAllHitsForFit( const ModSciFiHits::ModPrSciFiHitsSOA& allXHits, XCandidate& protoCand ) {
      for ( unsigned int iPlane{ 0 }; iPlane < Detector::FT::nXLayersTotal; ++iPlane ) {
        const auto idxSpan = protoCand.getIdxSpan( iPlane );
        std::transform( idxSpan.begin(), idxSpan.end(), std::back_inserter( protoCand.getCoordsToFit() ),
                        [&]( auto idx ) {
                          protoCand.xAtRef() += allXHits.coord( idx );
                          return allXHits.fulldex( idx );
                        } );
      }
      protoCand.xAtRef() /= static_cast<float>( protoCand.getCoordsToFit().size() );
    }

    /**
     * @brief fit linear parameters of parameterisation in x and remove bad hits
     * @param protoCand candidate to fit
     * @param pars bundle of parameters that tune hit removal
     * @param SciFiHits PrSciFiHits container
     * @details although only the linear parameters are fitted, the full parameterisation is used meaning
     * that the quadratic and the cubic term are fixed to their initial values.
     */
    template <bool secondLoop>
    void fitLinearXProjection( XCandidate& protoCand, const PrParametersX& pars, const FT::Hits& SciFiHits,
                               const VeloSeedExtended& veloSeed ) {

      auto& coordsToFit = protoCand.getCoordsToFit();
      bool  fit{ true };
      while ( fit ) {
        float s0{ 0.f }, sz{ 0.f }, sz2{ 0.f }, sd{ 0.f }, sdz{ 0.f };
        for ( auto iHit : coordsToFit ) {
          auto dz = SciFiHits.z( iHit ) - zReference;
          dz += veloSeed.yStraightInZone[SciFiHits.planeCode( iHit )] * SciFiHits.dzDy( iHit );
          const auto d = SciFiHits.x( iHit ) - protoCand.x( dz );
          const auto w = SciFiHits.w( iHit );
          s0 += w;
          sz += w * dz;
          sz2 += w * dz * dz;
          sd += w * d;
          sdz += w * d * dz;
        }
        const auto den = sz * sz - s0 * sz2;
        if ( essentiallyZero( den ) ) return;
        const auto da = ( sdz * sz - sd * sz2 ) / den;
        const auto db = ( sd * sz - s0 * sdz ) / den;
        protoCand.addXParams<2>( std::array{ da, db } );

        const auto itEnd = coordsToFit.end();
        auto       worst = itEnd;
        auto       maxChi2{ 0.f };
        const bool notMultiple = protoCand.nDifferentPlanes() == coordsToFit.size();
        for ( auto itH = coordsToFit.begin(); itH != itEnd; ++itH ) {
          auto dz = SciFiHits.z( *itH ) - zReference;
          dz += veloSeed.yStraightInZone[SciFiHits.planeCode( *itH )] * SciFiHits.dzDy( *itH );
          const auto d    = SciFiHits.x( *itH ) - protoCand.x( dz );
          const auto chi2 = d * d * SciFiHits.w( *itH );
          if ( chi2 > maxChi2 && ( notMultiple || protoCand.nInSamePlane( *itH ) > 1 ) ) {
            maxChi2 = chi2;
            worst   = itH;
          }
        }
        if ( worst == itEnd ) return;
        fit = false;
        if ( const int ip = SciFiHits.planeCode( *worst ) / 2u;
             maxChi2 > pars.maxChi2LinearFit || protoCand.planeSize( ip ) > 1 ) {
          protoCand.removePlane( ip );
          std::iter_swap( worst, std::prev( itEnd ) );
          coordsToFit.pop_back();
          if ( coordsToFit.size() < pars.minXHits ) return;
          fit = true;
        }
      }
    }

    /**
     * @brief try to find an x hit on a still empty x plane that matches the track
     * @param protoCand track candidate
     * @param pars bundle of parameters that tune the window defining if an x hit matches or not
     * @param veloSeed extended velo track
     * @param SciFiHits PrSciFiHits container
     * @param maxChi2XAdd defines the maximum allowed chi2 deviation from track
     * @details Hits are only added if they are within a window around the extrapolated position on the plane
     * under consideration and do not deviate too much in chi2.
     */
    template <bool secondLoop>
    auto fillEmptyXPlanes( XCandidate& protoCand, const PrParametersX& pars, const VeloSeedExtended& veloSeed,
                           const FT::Hits& SciFiHits, float maxChi2XAdd ) {

      if ( protoCand.nDifferentPlanes() == Detector::FT::nXLayersTotal ) return false;

      bool       added{ false };
      const auto xAtRef  = protoCand.getXParams()[0];
      const auto xWindow = pars.maxXWindow +
                           ( std::abs( xAtRef ) + std::abs( xAtRef - veloSeed.xStraightAtRef ) ) * pars.maxXWindowSlope;

      for ( unsigned int iPlane{ 0 }; iPlane < Detector::FT::nXLayersTotal; ++iPlane ) {
        if ( protoCand.planeSize( iPlane ) ) continue;
        const auto pc             = Detector::FT::xLayers[iPlane];
        const auto [xStart, xEnd] = SciFiHits.getZoneIndices( 2 * pc + veloSeed.upperHalfTrack );

        const auto xPred = protoCand.x( veloSeed.betterZ[pc] - zReference );

        const auto startwin = SciFiHits.get_lower_bound_fast<4>( xStart, xEnd, xPred - xWindow );
        auto       endwin{ startwin };
        auto       bestChi2{ maxChi2XAdd };
        auto       best{ 0 };
        for ( const auto maxX = xPred + xWindow; SciFiHits.x( endwin ) <= maxX; ++endwin ) {
          const auto d = SciFiHits.x( endwin ) - xPred;
          if ( const auto chi2 = d * d * SciFiHits.w( endwin ); chi2 < bestChi2 ) {
            bestChi2 = chi2;
            best     = endwin;
          }
        }
        if ( best ) {
          protoCand.getCoordsToFit().push_back( best );
          ++protoCand.nDifferentPlanes();
          ++protoCand.planeSize( iPlane );
          added = true;
        }
      }
      return added;
    }
    /**
     * @brief try to find an x hit on a still empty x plane that matches the track
     * @param track track candidate
     * @param pars bundle of parameters that tune the window defining if an x hit matches or not
     * @param veloSeed extended velo track
     * @param SciFiHits PrSciFiHits container
     * @param maxChi2XAdd defines the maximum allowed chi2 deviation from track
     * @details Hits are only added if they are within a window around the extrapolated position on the plane
     * under consideration and do not deviate too much in chi2. The empty planes are determined first.
     */
    template <bool secondLoop>
    auto fillEmptyXPlanes( PrForwardTrack& track, const PrParametersX& pars, const VeloSeedExtended& veloSeed,
                           const FT::Hits& SciFiHits, float maxChi2XAdd ) {

      if ( track.size() == Detector::FT::nLayersTotal ) return false;
      auto&                                    coordsToFit = track.getCoordsToFit();
      std::bitset<Detector::FT::nXLayersTotal> hasXLayer{};
      unsigned int                             nX{ 0 };
      for ( auto iHit : coordsToFit ) {
        if ( const auto pc = SciFiHits.planeCode( iHit ); Detector::FT::isXLayer[pc] ) {
          hasXLayer.set( pc / 2u );
          ++nX;
        }
      }
      if ( nX == Detector::FT::nXLayersTotal ) return false;
      bool       added{ false };
      const auto xAtRef  = track.getXParams()[0];
      const auto xWindow = pars.maxXWindow +
                           ( std::abs( xAtRef ) + std::abs( xAtRef - veloSeed.xStraightAtRef ) ) * pars.maxXWindowSlope;

      for ( unsigned int iPlane{ 0 }; iPlane < Detector::FT::nXLayersTotal; ++iPlane ) {
        if ( hasXLayer.test( iPlane ) ) continue;
        const auto pc             = Detector::FT::xLayers[iPlane];
        const auto [xStart, xEnd] = SciFiHits.getZoneIndices( 2 * pc + veloSeed.upperHalfTrack );

        const auto xPred = track.x( veloSeed.betterZ[pc] - zReference );

        const auto startwin = SciFiHits.get_lower_bound_fast<4>( xStart, xEnd, xPred - xWindow );
        auto       endwin{ startwin };
        auto       bestChi2{ maxChi2XAdd };
        auto       best{ 0 };
        for ( const auto maxX = xPred + xWindow; SciFiHits.x( endwin ) <= maxX; ++endwin ) {
          const auto d = SciFiHits.x( endwin ) - xPred;
          if ( const auto chi2 = d * d * SciFiHits.w( endwin ); chi2 < bestChi2 ) {
            bestChi2 = chi2;
            best     = endwin;
          }
        }
        if ( best ) {
          track.getCoordsToFit().push_back( best );
          added = true;
        }
      }
      return added;
    }

    /**
     * @brief fit up to quadratic parameters of parameterisation in x and remove bad hits
     * @param track candidate to fit
     * @param pars bundle of parameters that tune hit removal
     * @param SciFiHits PrSciFiHits container
     * @details although only the parameters up to the quadratic term are fitted, the full parameterisation is used
     * meaning that the cubic term is fixed to its initial value.
     */
    auto fitXProjection( XCandidate& track, const PrParametersX& pars, const FT::Hits& SciFiHits ) {
      auto&      coordsToFit = track.getCoordsToFit();
      const auto minHits     = pars.minXHits;
      bool       fit{ true };
      while ( fit ) {
        // plus one because we are fitting all params but one
        const auto nDoF = coordsToFit.size() - track.getXParams().size() + 1;
        if ( nDoF < 1 ) return false;
        auto s0{ 0.f }, sz{ 0.f }, sz2{ 0.f }, sz3{ 0.f }, sz4{ 0.f }, sd{ 0.f }, sdz{ 0.f }, sdz2{ 0.f };
        for ( auto iHit : coordsToFit ) {
          const auto dzNoScale = track.calcBetterDz( iHit );
          const auto d         = track.distanceXHit( iHit, dzNoScale );
          const auto w         = SciFiHits.w( iHit );
          const auto dz        = .001f * dzNoScale;
          const auto dz2       = dz * dz;
          const auto wdz       = w * dz;
          s0 += w;
          sz += wdz;
          sz2 += wdz * dz;
          sz3 += wdz * dz2;
          sz4 += wdz * dz2 * dz;
          sd += w * d;
          sdz += wdz * d;
          sdz2 += w * d * dz2;
        }
        const auto b1  = sz * sz - s0 * sz2;
        const auto c1  = sz2 * sz - s0 * sz3;
        const auto d1  = sd * sz - s0 * sdz;
        const auto b2  = sz2 * sz2 - sz * sz3;
        const auto c2  = sz3 * sz2 - sz * sz4;
        const auto d2  = sdz * sz2 - sz * sdz2;
        const auto den = b1 * c2 - b2 * c1;
        if ( essentiallyZero( den ) ) return false;
        const auto db = ( d1 * c2 - d2 * c1 ) / den;
        const auto dc = ( d2 * b1 - d1 * b2 ) / den;
        const auto da = ( sd - db * sz - dc * sz2 ) / s0;
        track.addXParams<3>( std::array{ da, db * 1.e-3f, dc * 1.e-6f } );

        auto maxChi2{ 0.f };
        auto totChi2{ 0.f };

        const auto itEnd = coordsToFit.end();
        auto       worst = itEnd;
        for ( auto itH = coordsToFit.begin(); itH != itEnd; ++itH ) {
          auto dz = SciFiHits.z( *itH ) - zReference;
          dz += track.y( dz ) * SciFiHits.dzDy( *itH );
          const auto chi2 = track.chi2XHit( *itH, dz );
          totChi2 += chi2;
          if ( chi2 > maxChi2 ) {
            maxChi2 = chi2;
            worst   = itH;
          }
        }
        track.setChi2NDoF( { totChi2, static_cast<float>( nDoF ) } );

        if ( worst == itEnd ) return true;

        fit = false;
        if ( totChi2 > pars.maxChi2PerDoF * nDoF || maxChi2 > pars.maxChi2XProjection ) {
          track.removeFromPlane( *worst );
          std::iter_swap( worst, std::prev( itEnd ) );
          coordsToFit.pop_back();
          if ( coordsToFit.size() < minHits ) return false;
          fit = true;
        }
      }
      return true;
    }

    /**
     * @brief fit up to quadratic parameters of parameterisation in x and remove bad hits
     * @param track candidate to fit
     * @param pars bundle of parameters that tune hit removal
     * @param SciFiHits PrSciFiHits container
     * @details although only the parameters up to the quadratic term are fitted, the full parameterisation is used
     * meaning that the cubic term is fixed to its initial value.
     */
    auto fitXProjection( PrForwardTrack& track, const PrParametersX& pars, const FT::Hits& SciFiHits ) {
      auto&      coordsToFit = track.getCoordsToFit();
      const auto minHits     = pars.minXHits + pars.minStereoHits;

      bool fit{ true };
      while ( fit ) {
        // plus one because we are fitting all params but one
        const auto nDoF = coordsToFit.size() - track.getXParams().size() + 1;
        if ( nDoF < 1 ) return false;
        auto s0{ 0.f }, sz{ 0.f }, sz2{ 0.f }, sz3{ 0.f }, sz4{ 0.f }, sd{ 0.f }, sdz{ 0.f }, sdz2{ 0.f };
        for ( auto iHit : coordsToFit ) {
          const auto dzNoScale = track.getBetterDz( iHit, SciFiHits.z( iHit ) - zReference, SciFiHits );
          const auto d         = track.distance( iHit, dzNoScale, SciFiHits );
          const auto w         = SciFiHits.w( iHit );
          const auto dz        = .001f * dzNoScale;
          const auto dz2       = dz * dz;
          const auto wdz       = w * dz;
          s0 += w;
          sz += wdz;
          sz2 += wdz * dz;
          sz3 += wdz * dz2;
          sz4 += wdz * dz2 * dz;
          sd += w * d;
          sdz += wdz * d;
          sdz2 += w * d * dz2;
        }
        const auto b1  = sz * sz - s0 * sz2;
        const auto c1  = sz2 * sz - s0 * sz3;
        const auto d1  = sd * sz - s0 * sdz;
        const auto b2  = sz2 * sz2 - sz * sz3;
        const auto c2  = sz3 * sz2 - sz * sz4;
        const auto d2  = sdz * sz2 - sz * sdz2;
        const auto den = b1 * c2 - b2 * c1;
        if ( essentiallyZero( den ) ) return false;
        const auto db = ( d1 * c2 - d2 * c1 ) / den;
        const auto dc = ( d2 * b1 - d1 * b2 ) / den;
        const auto da = ( sd - db * sz - dc * sz2 ) / s0;
        track.addXParams<3>( std::array{ da, db * 1.e-3f, dc * 1.e-6f } );

        auto maxChi2{ 0.f };
        auto totChi2{ 0.f };

        const auto itEnd = coordsToFit.end();
        auto       worst = itEnd;
        for ( auto itH = coordsToFit.begin(); itH != itEnd; ++itH ) {
          const auto dz   = track.getBetterDz( *itH, SciFiHits.z( *itH ) - zReference, SciFiHits );
          const auto chi2 = track.chi2( *itH, dz, SciFiHits );
          totChi2 += chi2;
          if ( chi2 > maxChi2 ) {
            maxChi2 = chi2;
            worst   = itH;
          }
        }
        track.setChi2NDoF( { totChi2, static_cast<float>( nDoF ) } );

        if ( worst == itEnd ) return true;

        fit = false;
        if ( totChi2 > pars.maxChi2PerDoF * nDoF || maxChi2 > pars.maxChi2XProjection ) {
          std::iter_swap( worst, std::prev( itEnd ) );
          coordsToFit.pop_back();
          if ( coordsToFit.size() < minHits ) return false;
          fit = true;
        }
      }
      return true;
    }

    /**
     * @brief Collect the stereo layer hits that are on the other side because of triangle reaching into side.
     *
     * @param zoneNumberOS zoneNumber of the other side
     * @param xMin minimum coordinate to check on this layer
     * @param xMax maximum coordinate to check on this layer
     * @param dxDySign the sign of stereo rotation of layer
     * @param xPredShifted predicated coordinate corrected for shift due to stereo angle
     * @param yInZone y position of track in layer
     * @param SciFiHits hits in detector
     * @param pars y parameter pack
     * @param hough instance of HoughSerach
     */
    [[gnu::noinline]] void collectTriangleHits( unsigned zoneNumberOS, float xMin, float xMax, float dxDySign,
                                                float xPredShifted, float yInZone, const FT::Hits& SciFiHits,
                                                const PrParametersY& pars, StereoSearch& hough ) {

      const auto [uvStart, uvEnd] = SciFiHits.getZoneIndices( zoneNumberOS );
      const auto iUVStart         = SciFiHits.get_lower_bound_fast<4>( uvStart, uvEnd, xMin );
      const auto yMin             = yInZone + pars.yTolUVSearch;
      const auto yMax             = yInZone - pars.yTolUVSearch;
      auto       triangleOK       = [&]( int index ) {
        const auto [yMinHit, yMaxHit] = SciFiHits.yEnd( index );
        return yMax < yMaxHit && yMin > yMinHit;
      };
      for ( auto iUV{ iUVStart }; SciFiHits.x( iUV ) <= xMax; ++iUV ) {
        const auto signedDx = ( SciFiHits.x( iUV ) - xPredShifted ) * dxDySign;
        if ( triangleOK( iUV ) ) { hough.add( zoneNumberOS / 4u, signedDx, iUV ); }
      }
    }

    /**
     * @brief Collect stereo hits close to the x projection of the track in all layers.
     *
     * @param search_result this is the container where the candidates from the search are written to
     * @param track the track with mostly x information yet
     * @param SciFiHits hits in the detector
     * @param veloSeed velo track parameters
     * @param cache cache for position of scifi zones
     * @param pars parameter pack for stereo candidate
     * @return gsl::span span over candidates
     */
    auto collectStereoHits( StereoSearch::result_type& search_result, const PrForwardTrack& track,
                            const FT::Hits& SciFiHits, const VeloSeedExtended& veloSeed, const ZoneCache& cache,
                            const PrParametersY& pars ) {

      const auto minTol = -pars.tolY - pars.tolYSlope * ( HoughTransformation::rangeMax + pars.maxTolY );
      // the first parameter controls the threshold which bins in the hough search have to have at least
      StereoSearch hough{ Detector::FT::nUVLayersTotal / 2, minTol, pars.uvSearchBinWidth };
      direct_debug( "Collect stereo hits with maximal", std::abs( minTol ), "mm deviation from x track." );

      const auto& uvZones = veloSeed.upperHalfTrack ? Detector::FT::uvZonesUpper : Detector::FT::uvZonesLower;
      for ( auto zoneNumber : uvZones ) {
        const auto side  = track.x( cache.z( zoneNumber / 2u ) - zReference ) > 0.f
                               ? LHCb::Detector::FTChannelID::Side::A
                               : LHCb::Detector::FTChannelID::Side::C;
        const auto zZone = cache.z( zoneNumber, side );
        // TODO: can we improve here by using the ShiftCalculator?
        const auto yInZone      = track.y( zZone - zReference );
        const auto betterZ      = zZone + yInZone * cache.dzdy( zoneNumber, side );
        const auto xPred        = track.x( betterZ - zReference );
        const auto dxDy         = cache.dxdy( zoneNumber, side );
        const auto xPredShifted = xPred - yInZone * dxDy;
        const auto dxTol =
            pars.tolY +
            pars.tolYSlope * ( std::abs( xPred - veloSeed.xStraightInZone[zoneNumber / 2u] ) + std::abs( yInZone ) );
        const auto [uvStart, uvEnd] = SciFiHits.getZoneIndices( zoneNumber );
        const auto xMin             = xPredShifted - dxTol;
        const auto xMax             = xPredShifted + dxTol;
        // the difference only is interepreation: currently: negative dx * sign means underestimated y position and vice
        // versa but it makes a difference for the sorting!
        const auto dxDySign = std::copysign( 1.f, dxDy );
        const auto iUVStart = SciFiHits.get_lower_bound_fast<4>( uvStart, uvEnd, xMin );
        for ( auto iUV{ iUVStart }; SciFiHits.x( iUV ) <= xMax; ++iUV ) {
          const auto dz          = SciFiHits.z( iUV ) + yInZone * SciFiHits.dzDy( iUV ) - zReference;
          const auto predShifted = track.x( dz ) - yInZone * SciFiHits.dxDy( iUV );
          const auto signedDx    = ( SciFiHits.x( iUV ) - predShifted ) * dxDySign;
          // we actually only care about 0-5 for layers, so divide by 4
          hough.add( zoneNumber / 4u, signedDx, iUV );
        }
        if ( std::abs( yInZone ) < pars.tolYTriangleSearch ) {
          const auto zoneNumberOS = veloSeed.upperHalfTrack ? zoneNumber - 1 : zoneNumber + 1;
          collectTriangleHits( zoneNumberOS, xMin, xMax, dxDySign, xPredShifted, yInZone, SciFiHits, pars, hough );
        }
        direct_debug( "stereoLayer", zoneNumber / 2u, "| betterZ =", betterZ, "| xShiftAtY =", yInZone * dxDy,
                      "| xPred =", xPred, "| [xMin,xMax] (index) = [", xMin, "(", iUVStart, "),", xMax, "]" );
      }
      // the search takes into account bins that are above threshold and additionally checks that with
      // both neighbours more than minStereoHits layers are present
      // FIXME: this probably sometimes finds identical candidates when two neighboursing bins have e.g. 3 layers
      const auto last = hough.search( search_result.begin(), pars.minStereoHits );
      return LHCb::make_span( search_result.begin(), last );
    }

    /**
     * @brief fit linear parameters of parameterisation in y and remove bad hits
     * @param track candidate to fit
     * @param pars bundle of parameters that tune hit removal
     * @param SciFiHits PrSciFiHits container
     * @details although only the linear parameters are fitted, the full parameterisation is used meaning
     * that the quadratic term is fixed to its initial values.
     */
    auto fitLinearYProjection( StereoCandidate& track, const PrParametersY& pars, const FT::Hits& SciFiHits ) {

      auto& coordsToFit = track.getCoordsToFit();
      auto  fit{ true };
      while ( fit ) {
        auto s0{ 0.f }, sz{ 0.f }, sz2{ 0.f }, sd{ 0.f }, sdz{ 0.f };
        for ( auto iHit : coordsToFit ) {
          const auto dz = track.calcBetterDz( iHit, SciFiHits );
          const auto d  = track.distanceStereoHit( iHit, dz, SciFiHits );
          const auto w  = SciFiHits.w( iHit );
          s0 += w;
          sz += w * dz;
          sz2 += w * dz * dz;
          sd += w * d;
          sdz += w * d * dz;
        }
        const auto den = s0 * sz2 - sz * sz;
        if ( essentiallyZero( den ) ) return false;
        const auto da = ( sd * sz2 - sdz * sz ) / den;
        const auto db = ( sdz * s0 - sd * sz ) / den;
        track.addYParams<2>( std::array{ da, db } );

        const auto itEnd = coordsToFit.end();
        auto       worst = itEnd;
        auto       maxChi2{ 0.f };
        for ( auto itH = coordsToFit.begin(); itH != itEnd; ++itH ) {
          const auto chi2 = track.chi2StereoHits( *itH, SciFiHits );
          if ( chi2 > maxChi2 ) {
            maxChi2 = chi2;
            worst   = itH;
          }
        }
        if ( worst == itEnd ) return true;
        fit = false;
        if ( maxChi2 > pars.maxChi2StereoLinear ) {
          if ( coordsToFit.size() - 1 < pars.minStereoHits ) return false;
          track.removeFromPlane( *worst, SciFiHits );
          std::iter_swap( worst, std::prev( itEnd ) );
          coordsToFit.pop_back();
          fit = true;
        }
      }
      return true;
    }

    /**
     * @brief fit parameters of parameterisation in y and remove bad hits
     * @param track candidate to fit
     * @param pars bundle of parameters that tune hit removal
     * @param SciFiHits PrSciFiHits container
     * @param veloSeed velo track extension
     * @details The fit uses a contraint to the y position at the magnet kink z position. Since there's already
     * an estimate of the x parameters, an improved estimate of the magnet kink z position is used. The weight of
     * this position in the fit is much smaller than the one of hits.
     */
    auto fitYProjection( StereoCandidate& track, const PrParametersY& pars, const FT::Hits& SciFiHits,
                         const VeloSeedExtended& veloSeed ) {

      const auto dSlope    = track.getXParams()[1] - veloSeed.seed.tx;
      const auto zMagMatch = veloSeed.calcZMag( dSlope );

      const auto tolYMag     = pars.tolYMag + pars.tolYMagSlope * std::abs( dSlope );
      const auto wMag        = 1.f / ( tolYMag * tolYMag );
      const auto dzMagRef    = .001f * ( zMagMatch - zReference );
      auto&      coordsToFit = track.getCoordsToFit();

      bool fit{ true };
      while ( fit ) {
        assert( pars.minStereoHits >= coordsToFit.size() - track.getYParams().size() );
        const auto dyMag = track.yStraight( zMagMatch ) - veloSeed.seed.y( zMagMatch );
        auto       s0    = wMag;
        auto       sz    = wMag * dzMagRef;
        auto       sz2   = wMag * dzMagRef * dzMagRef;
        auto       sd    = wMag * dyMag;
        auto       sdz   = wMag * dyMag * dzMagRef;
        auto       sz2m{ 0.f }, sz3{ 0.f }, sz4{ 0.f }, sdz2{ 0.f };

        for ( auto iHit : coordsToFit ) {
          const auto dzNoScale = track.calcBetterDz( iHit, SciFiHits );
          const auto d         = track.distanceStereoHit( iHit, dzNoScale, SciFiHits );
          const auto w         = SciFiHits.w( iHit );
          const auto dz        = .001f * dzNoScale;
          const auto dzdz      = dz * dz;
          const auto wdz       = w * dz;
          s0 += w;
          sz += wdz;
          sz2m += w * dzdz;
          sz2 += w * dzdz;
          sz3 += wdz * dzdz;
          sz4 += wdz * dz * dzdz;
          sd += w * d;
          sdz += wdz * d;
          sdz2 += wdz * d * dz;
        }
        const auto b1  = sz * sz - s0 * sz2;
        const auto c1  = sz2m * sz - s0 * sz3;
        const auto d1  = sd * sz - s0 * sdz;
        const auto b2  = sz2 * sz2m - sz * sz3;
        const auto c2  = sz3 * sz2m - sz * sz4;
        const auto d2  = sdz * sz2m - sz * sdz2;
        const auto den = b1 * c2 - b2 * c1;
        if ( essentiallyZero( den ) ) return false;
        const auto db = ( d1 * c2 - d2 * c1 ) / den;
        const auto dc = ( d2 * b1 - d1 * b2 ) / den;
        const auto da = ( sd - db * sz - dc * sz2 ) / s0;
        track.addYParams<3>( std::array{ da, db * 1.e-3f, dc * 1.e-6f } );

        const auto itEnd = coordsToFit.end();
        auto       worst = itEnd;
        auto       maxChi2{ 0.f };
        for ( auto itH = coordsToFit.begin(); itH != itEnd; ++itH ) {
          const auto chi2 = track.chi2StereoHits( *itH, SciFiHits );
          if ( chi2 > maxChi2 ) {
            maxChi2 = chi2;
            worst   = itH;
          }
        }
        if ( worst == itEnd ) return true;
        fit = false;
        if ( maxChi2 > pars.maxChi2Stereo ) {
          if ( coordsToFit.size() - 1 < pars.minStereoHits ) return false;
          track.removeFromPlane( *worst, SciFiHits );
          std::iter_swap( worst, std::prev( itEnd ) );
          coordsToFit.pop_back();
          fit = true;
        }
      }
      return true;
    }

    /**
     * @brief check for hits in triangle zones
     */
    [[gnu::noinline]] auto checkTriangleOnEmptyPlane( int iPlane, int zoneNumberOS, float xMin, float xMax,
                                                      float xPredShifted, float yInZone, StereoCandidate& track,
                                                      const FT::Hits& SciFiHits, const PrParametersY& pars ) {

      const auto [uvStart, uvEnd] = SciFiHits.getZoneIndices( zoneNumberOS );
      const auto iUVStart         = SciFiHits.get_lower_bound_fast<4>( uvStart, uvEnd, xMin );
      const auto yMin             = yInZone + pars.yTolUVSearch;
      const auto yMax             = yInZone - pars.yTolUVSearch;
      auto       triangleOK       = [&]( int index ) {
        const auto [yMinHit, yMaxHit] = SciFiHits.yEnd( index );
        return yMax < yMaxHit && yMin > yMinHit;
      };
      auto bestChi2{ pars.maxChi2Stereo };
      auto best{ 0 };
      for ( auto iUV{ iUVStart }; SciFiHits.x( iUV ) <= xMax; ++iUV ) {
        const auto d = SciFiHits.x( iUV ) - xPredShifted;
        if ( const auto chi2 = d * d * SciFiHits.w( iUV ); chi2 < bestChi2 && triangleOK( iUV ) ) {
          bestChi2 = chi2;
          best     = iUV;
        }
      }
      if ( best ) {
        track.getCoordsToFit().push_back( best );
        track.setPlaneUsed( iPlane );
        return true;
      }
      return false;
    }

    /**
     * @brief try to find an stereo hit on a still empty stereo plane that matches the track
     * @param track track candidate
     * @param SciFiHits PrSciFiHits container
     * @param veloSeed extended velo track
     * @param cache cache for zone properties
     * @param pars bundle of parameters that tune the window defining if an stereo hit matches or not
     * @details Hits are only added if they are within a window around the extrapolated position on the plane
     * under consideration and do not deviate too much in chi2. If the track is close to the triangle zones,
     * these are also checked for hits.
     */
    auto fillEmptyStereoPlanes( StereoCandidate& track, const FT::Hits& SciFiHits, const VeloSeedExtended& veloSeed,
                                const ZoneCache& cache, const PrParametersY& pars ) {

      if ( track.nDifferentPlanes() == Detector::FT::nUVLayersTotal ) return false;
      bool added{ false };
      for ( unsigned int iPlane{ 0 }; iPlane < Detector::FT::nUVLayersTotal; ++iPlane ) {
        if ( track.isPlaneUsed( iPlane ) ) continue;
        const auto pc           = Detector::FT::stereoLayers[iPlane];
        const auto zoneNumber   = 2 * pc + veloSeed.upperHalfTrack;
        const auto side         = track.x( cache.z( pc ) - zReference ) > 0.f ? LHCb::Detector::FTChannelID::Side::A
                                                                              : LHCb::Detector::FTChannelID::Side::C;
        const auto zZone        = cache.z( zoneNumber, side );
        const auto betterZ      = zZone + track.y( zZone - zReference ) * cache.dzdy( zoneNumber, side );
        const auto yInZone      = track.y( betterZ - zReference );
        const auto xPred        = track.x( betterZ - zReference );
        const auto xPredShifted = xPred - yInZone * cache.dxdy( zoneNumber, side );
        // TODO: here and in collection step is it better to use xPredShifted?
        const auto dxTol =
            pars.tolY + pars.tolYSlope * ( std::abs( xPred - veloSeed.xStraightInZone[pc] ) + std::abs( yInZone ) );
        const auto [uvStart, uvEnd] = SciFiHits.getZoneIndices( zoneNumber );
        const auto xMin             = xPredShifted - dxTol;
        const auto xMax             = xPredShifted + dxTol;
        const auto iUVStart         = SciFiHits.get_lower_bound_fast<4>( uvStart, uvEnd, xMin );
        auto       bestChi2{ pars.maxChi2StereoAdd };
        auto       best{ 0 };
        for ( auto iUV{ iUVStart }; SciFiHits.x( iUV ) <= xMax; ++iUV ) {
          const auto dz          = SciFiHits.z( iUV ) + yInZone * SciFiHits.dzDy( iUV ) - zReference;
          const auto predShifted = track.x( dz ) - yInZone * SciFiHits.dxDy( iUV );
          const auto d           = SciFiHits.x( iUV ) - predShifted;
          if ( const auto chi2 = d * d * SciFiHits.w( iUV ); chi2 < bestChi2 ) {
            bestChi2 = chi2;
            best     = iUV;
          }
        }
        if ( best ) {
          track.getCoordsToFit().push_back( best );
          track.setPlaneUsed( iPlane );
          added = true;
        }
        if ( !added && std::abs( yInZone ) < pars.tolYTriangleSearch ) {
          const auto zoneNumberOS = veloSeed.upperHalfTrack ? zoneNumber - 1 : zoneNumber + 1;
          added = checkTriangleOnEmptyPlane( iPlane, zoneNumberOS, xMin, xMax, xPredShifted, yInZone, track, SciFiHits,
                                             pars );
        }
      }
      return added;
    }

    /**
     * @brief stereo hit selection for given x candidate
     * @details The idea is to check for stereo hits that have a similar deviation form the x track. This takes into
     * account that the y position is not well known and thus often over- or underestimated which leads to similar
     * systematic shifts of the stereo hits in x and hence similar distances to the x track. The hits belonging together
     * are found using a Hough cluser search similar to the one to find initial x candidates. However, because we
     * already have an x candidate here the number of hits to consider is smaller compared to the x candidate
     * combinatorics. Thus the Hough search is simpler and taken from the generic implementation in HoughSearch.h. A
     * best set of stereo hits is selected by using the total chi2 of the y projection fit and the total number of
     * stereo hits found.
     */
    auto selectStereoHits( PrForwardTrack& track, const FT::Hits& SciFiHits, const PrParametersY& yPars,
                           const ZoneCache& cache, const VeloSeedExtended& veloSeed ) {

      static_vector<int, Detector::FT::nUVLayersTotal> bestStereoHits{};
      TrackPars<3>                                     bestYParams{};
      auto                                             bestMeanDy2{ std::numeric_limits<float>::max() };
      StereoCandidate                                  stereoCand{};
      StereoSearch::result_type                        combinations;
      stereoCand.getXParams() = track.getXParams();

      for ( auto& combo : collectStereoHits( combinations, track, SciFiHits, veloSeed, cache, yPars ) ) {
        stereoCand.clear();
        stereoCand.getYParams() = track.getYParams();
        for ( auto iHit : combo ) {
          if ( StereoSearch::invalid( iHit ) ) break;
          stereoCand.setPlaneUsed( SciFiHits.planeCode( iHit ) / 2u );
          stereoCand.getCoordsToFit().push_back( iHit );
        }
        direct_debug( "" );
        direct_debug( "StereoCandidate before fit      :", stereoCand );

        if ( !fitLinearYProjection( stereoCand, yPars, SciFiHits ) ) continue;
        direct_debug( "StereoCandidate after linear fit:", stereoCand );

        if ( !fitYProjection( stereoCand, yPars, SciFiHits, veloSeed ) ) continue;

        if ( fillEmptyStereoPlanes( stereoCand, SciFiHits, veloSeed, cache, yPars ) ) {
          if ( !fitYProjection( stereoCand, yPars, SciFiHits, veloSeed ) ) continue;
        }

        if ( auto& candidateHits = stereoCand.getCoordsToFit(); candidateHits.size() >= bestStereoHits.size() ) {
          const auto meanDy2 = std::accumulate( candidateHits.begin(), candidateHits.end(), 0.f,
                                                [&]( auto init, auto iHit ) {
                                                  const auto dz = stereoCand.calcBetterDz( iHit, SciFiHits );
                                                  const auto d  = stereoCand.distanceStereoHit( iHit, dz, SciFiHits );
                                                  return init + d * d;
                                                } ) /
                               static_cast<float>( candidateHits.size() - 1 );
          if ( candidateHits.size() > bestStereoHits.size() || meanDy2 < bestMeanDy2 ) {
            bestYParams = stereoCand.getYParams();
            bestMeanDy2 = meanDy2;
            bestStereoHits.clear();
            std::copy( std::make_move_iterator( candidateHits.begin() ), std::make_move_iterator( candidateHits.end() ),
                       std::back_inserter( bestStereoHits ) );
            direct_debug( "Best StereoCandidate:", stereoCand, "meanDy2 =", meanDy2 );
          }
        }
        direct_debug( "" );
      }
      if ( !bestStereoHits.empty() ) {
        track.getYParams() = std::move( bestYParams );
        track.addHits( std::move( bestStereoHits ) );
        return true;
      }
      return false;
    }

    /**
     * @brief calculates distance in x between input track and forward track at magnet kink
     */
    auto calcXMatchDistance( const PrForwardTrack& track, const VeloSeedExtended& veloSeed ) {
      constexpr auto zEndT     = static_cast<float>( StateParameters::ZEndT );
      constexpr auto dz        = zEndT - zReference;
      const auto     xEndT     = track.x( dz );
      const auto     txEndT    = track.xSlope( dz );
      const auto     tx        = veloSeed.seed.tx;
      const auto     dSlope    = track.getXParams()[1] - tx;
      const auto     zMagMatch = veloSeed.calcZMag( dSlope );
      return std::abs( veloSeed.seed.x( zMagMatch ) - ( xEndT + ( zMagMatch - zEndT ) * txEndT ) );
    }

    /**
     * @brief calculates distance in y between input track and forward track behind the last SciFi station
     */
    auto calcYMatchDistance( const PrForwardTrack& track, const VeloSeedExtended& veloSeed ) {
      constexpr auto zEndT   = static_cast<float>( StateParameters::ZEndT );
      constexpr auto dz      = zEndT - zReference;
      const auto     dSlope  = track.xSlope( dz ) - veloSeed.seed.tx;
      const auto     dSlopeY = track.ySlope( dz ) - veloSeed.seed.ty;
      const auto     yAtEndT = veloSeed.seed.y( zEndT ) + veloSeed.veloSciFiMatch.calcYCorr( dSlope, dSlopeY );
      return yAtEndT - track.y( dz );
    }

    /**
     * @brief Main work of the Hough transformation: projecting hits to "Hough Space".
     * @param veloSeed Contains useful information about the input track.
     * @param momentumBorders Only hits within a search window are projected, defined by this pair of bounds.
     * @details This method is very heavy and should be as optimised as possible. It calculates the projection of
     * each x hit on each SciFi layer within the serach window to the reference plane. For each hit a bin number is
     * calculated and the necessary information about the hit is stored in the containers of the histogram.
     */
    void HoughTransformation::projectXHitsToHoughSpace( const VeloSeedExtended& veloSeed,
                                                        std::pair<float, float> momentumBorders,
                                                        const FT::Hits&         SciFiHits ) {
      direct_debug( "------------projectXHitsToHoughSpace------------" );
      direct_debug( "" );
      const auto dxMin    = momentumBorders.first;
      const auto dxMax    = momentumBorders.second;
      const auto dzInvRef = 1.f / ( zReference - veloSeed.zMag );

      for ( const auto iLayer : Detector::FT::xLayers ) {
        const auto zZone           = veloSeed.betterZ[iLayer];
        const auto xStraightInZone = veloSeed.xStraightInZone[iLayer];
        const auto step            = dzInvRef * ( zZone - veloSeed.zMag );
        const auto xMin            = xStraightInZone + dxMin * step;
        const auto xMax            = xStraightInZone + dxMax * step;
        // lower zone numbers are always even, corresponding upper half number has + 1, i.e. add boolean
        // get zone indices from hit container
        auto [minIdx, maxIdx] = SciFiHits.getZoneIndices( 2 * iLayer + veloSeed.upperHalfTrack );

        // binary search to find indicies corresponding to search window
        minIdx = SciFiHits.get_lower_bound_fast<2>( minIdx, maxIdx, xMin );
        maxIdx = SciFiHits.get_lower_bound_fast<4>( minIdx, maxIdx, xMax );
        direct_debug( "xLayer", iLayer, "| betterZ =", zZone, "| [xMin,xMax] (index) = [", xMin, "(", minIdx, "),",
                      xMax, "(", maxIdx, ")]" );
        const auto tx            = veloSeed.seed.tx;
        const auto dSlopeDivPart = 1.f / ( zZone - veloSeed.zMag );
        const auto dz            = zZone - zReference;
        const auto dz2           = dz * dz;
        const auto x0            = xStraightInZone - tx * zZone;
        // encode plane in bits above first nDiffPlanesBits (8) bits, this will be used to check if plane is already
        // used
        const auto encodedPlaneNumber = 1u << ( iLayer + nDiffPlanesBits );

        for ( auto i{ minIdx }; i < maxIdx; i += simd::size ) {
          const auto hits        = SciFiHits.x<simd::float_v>( i );
          const auto dSlope      = ( hits - xStraightInZone ) * dSlopeDivPart;
          const auto zMagPrecise = veloSeed.calcZMag( dSlope );
          const auto xMag        = x0 + tx * zMagPrecise;
          const auto ratio       = ( zReference - zMagPrecise ) / ( zZone - zMagPrecise );
          const auto xCorrection = veloSeed.calcXCorr( dSlope, dz, dz2 );
          // position of hits is moved by a correction to account for curvature
          const auto projHit         = xMag + ratio * ( hits - xMag - xCorrection );
          const auto binNumber       = calculateBinNumber( projHit, i, maxIdx );
          const auto encodedPlaneCnt = gather( m_planeCounter.data(), binNumber );
          // OR with current plane to set flag if plane is not present in the bin yet
          const auto encodedPlaneCntNewFlag = encodedPlaneCnt | encodedPlaneNumber;
          // remember: the first nDiffPlanesBits (8) encode the number of different planes
          // so let's simply check if we added a flag, and if so, add 1 to number of diff planes
          const auto newEncodedPlaneCnt = encodedPlaneCntNewFlag + ( encodedPlaneCnt < encodedPlaneCntNewFlag );

          const auto binNumberBase = SIMDWrapper::to_array( binNumber );
          const auto planeCntBase  = SIMDWrapper::to_array( newEncodedPlaneCnt );
          const auto projHitBase   = SIMDWrapper::to_array( projHit );
          // the bins are static, always check that they do not overflow
          for ( size_t idx{ 0 }; idx < simd::size; ++idx ) {
            if ( const auto iBin = binNumberBase[idx]; m_binContentSize[iBin] < reservedBinContent ) {
              m_planeCounter[iBin]         = planeCntBase[idx];
              const auto iBinIdx           = reservedBinContent * iBin + m_binContentSize[iBin]++;
              m_binContentCoord[iBinIdx]   = projHitBase[idx];
              m_binContentFulldex[iBinIdx] = i + idx;
            } else {
              direct_debug( "Bin", iBin, "is overflowing!" );
            }
          }
        } // end of loop over hits
        // bin 0 is the garbage bin, it contains out of range hits, let's empty it
        m_binContentSize[0] = 0;
        debug_tuple( [&, i = minIdx, maxIdx = maxIdx] {
          const auto hits        = SciFiHits.x<simd::float_v>( i );
          const auto dSlope      = ( hits - xStraightInZone ) * dSlopeDivPart;
          const auto zMagPrecise = veloSeed.calcZMag( dSlope );
          const auto xMag        = xStraightInZone + tx * ( zMagPrecise - zZone );
          const auto ratio       = ( zReference - zMagPrecise ) / ( zZone - zMagPrecise );
          const auto xCorrection = veloSeed.calcXCorr( dSlope, dz, dz2 );
          // position of hits is moved by a correction to account for curvature
          const auto projHit   = xMag + ratio * ( hits - xCorrection - xMag );
          const auto binNumber = calculateBinNumber( projHit, i, maxIdx );
          return std::make_tuple( "hit =", hits, "zMagPrecise =", zMagPrecise, "hit@ref =", projHit,
                                  "binNumber =", binNumber );
        }() );
        direct_debug( "" );
      } // end of loop over zones
    }

    /**
     * @brief Main work of the Hough transformation: projecting hits to "Hough Space".
     * @param veloSeed Contains useful information about the input track.
     * @param cache cache for zone properties
     * @param momentumBorders Only hits within a search window are projected, defined by this pair of bounds.
     * @details This method is very heavy and should be as optimised as possible. It calculates the projection of
     * each stereo hit on each SciFi layer within the serach window to the reference plane. For each hit a bin number is
     * calculated and the necessary information about the hit is stored in the containers of the histogram.
     */
    void HoughTransformation::projectStereoHitsToHoughSpace( const VeloSeedExtended& veloSeed, const ZoneCache& cache,
                                                             std::pair<float, float> momentumBorders,
                                                             const FT::Hits&         SciFiHits ) {
      direct_debug( "------------projectStereoHitsToHoughSpace------------" );
      direct_debug( "" );
      const auto dxMin    = momentumBorders.first;
      const auto dxMax    = momentumBorders.second;
      const auto dzInvRef = 1.f / ( zReference - veloSeed.zMag );

      for ( const auto iLayer : Detector::FT::stereoLayers ) {
        const auto zZone = veloSeed.betterZ[iLayer];
        // dxdy is what makes stereo hits shifty
        const auto dxDy            = cache.dxdy( iLayer );
        const auto xShiftAtY       = veloSeed.seed.y( zZone ) * dxDy;
        const auto xStraightInZone = veloSeed.xStraightInZone[iLayer];
        const auto step            = dzInvRef * ( zZone - veloSeed.zMag );
        const auto xMin            = xStraightInZone + dxMin * step - xShiftAtY;
        const auto xMax            = xStraightInZone + dxMax * step - xShiftAtY;

        auto [minIdx, maxIdx] = SciFiHits.getZoneIndices( 2 * iLayer + veloSeed.upperHalfTrack );

        minIdx = SciFiHits.get_lower_bound_fast<2>( minIdx, maxIdx, xMin );
        maxIdx = SciFiHits.get_lower_bound_fast<4>( minIdx, maxIdx, xMax );
        direct_debug( "stereoLayer", iLayer, "| betterZ =", zZone, "| xShiftAtY =", xShiftAtY,
                      "| [xMin,xMax] (index) = [", xMin, "(", minIdx, "),", xMax, "(", maxIdx, ")]" );
        const auto tx                 = veloSeed.seed.tx;
        const auto dSlopeDivPart      = 1.f / ( zZone - veloSeed.zMag );
        const auto dz                 = zZone - zReference;
        const auto dz2                = dz * dz;
        const auto x0                 = xStraightInZone - tx * zZone;
        const auto encodedPlaneNumber = 1u << ( iLayer + nDiffPlanesBits );
        for ( auto i{ minIdx }; i < maxIdx; i += simd::size ) {
          // first shift hits by total shift determined by straight line y extrapolation
          auto       hits        = SciFiHits.x<simd::float_v>( i ) + xShiftAtY;
          const auto dSlope      = ( hits - xStraightInZone ) * dSlopeDivPart;
          const auto zMagPrecise = veloSeed.calcZMag( dSlope );
          // then correct shift using parameterisation
          hits             = hits + veloSeed.calcYCorr( dSlope, iLayer ) * dxDy - veloSeed.calcXCorr( dSlope, dz, dz2 );
          const auto xMag  = x0 + tx * zMagPrecise;
          const auto ratio = ( zReference - zMagPrecise ) / ( zZone - zMagPrecise );
          const auto projHit                = xMag + ratio * ( hits - xMag );
          const auto binNumber              = calculateBinNumber( projHit, i, maxIdx );
          const auto encodedPlaneCnt        = gather( m_planeCounter.data(), binNumber );
          const auto encodedPlaneCntNewFlag = encodedPlaneCnt | encodedPlaneNumber;
          const auto newEncodedPlaneCnt     = encodedPlaneCntNewFlag + ( encodedPlaneCnt < encodedPlaneCntNewFlag );
          scatter( m_planeCounter.data(), binNumber, newEncodedPlaneCnt );
        } // end of loop over hits

        debug_tuple( [&, i = minIdx, maxIdx = maxIdx] {
          auto       hits   = SciFiHits.x<simd::float_v>( i ) + xShiftAtY;
          const auto dSlope = ( hits - xStraightInZone ) * dSlopeDivPart;
          // then correct shift using parameterisation
          hits                   = hits + veloSeed.calcYCorr( dSlope, iLayer ) * dxDy;
          const auto zMagPrecise = veloSeed.calcZMag( dSlope );
          const auto xMag        = xStraightInZone + tx * ( zMagPrecise - zZone );
          const auto ratio       = ( zReference - zMagPrecise ) / ( zZone - zMagPrecise );
          const auto xCorrection = veloSeed.calcXCorr( dSlope, dz, dz2 );
          const auto projHit     = xMag + ratio * ( hits - xCorrection - xMag );
          const auto binNumber   = calculateBinNumber( projHit, i, maxIdx );
          return std::make_tuple( "hit =", hits, "zMagPrecise =", zMagPrecise, "hit@ref =", projHit,
                                  "binNumber =", binNumber );
        }() );
        direct_debug( "" );
      } // end of loop over zones
    }

    /**
     * @brief Sort each picked up candidate bin and copy content to modifiable hit container.
     * @param neighbourCheck number of different planes in bin up to which both direct neighbours are also sorted and
     * copied
     * @param allXHits the container to copy to
     * @details Checking of neighbours makes it necessary to take care of not sorting and selecting a bin twice. This is
     * simply done by storing the last two copied bin numbers. The start and end indices of each candidates are stored
     * for direct candidate access later. If there's an overlap between different candidates' bins, the start and end
     * index is removed which means that the index range for both candidates is merged.
     */
    void HoughTransformation::sortAndCopyBinContents( int neighbourCheck, ModSciFiHits::ModPrSciFiHitsSOA& allXHits ) {

      auto lastBinCopied{ 0 };
      auto last2ndBinCopied{ 0 };
      for ( auto iCand{ 0 }; iCand < m_candSize; ++iCand ) {
        const auto iBin = m_candidateBins[iCand];
        assert( iBin );
        const int nDiff = decodeNbDifferentPlanes<scalar::int_v>( iBin ).cast();
        // TODO: also check neighbours if two times more hits than planes or so!
        const auto checkNeighbours = nDiff <= neighbourCheck;
        allXHits.candidateStartIndex.push_back( allXHits.size() );
        if ( checkNeighbours ) {
          if ( iBin - 1 != lastBinCopied && iBin - 1 != last2ndBinCopied ) {
            sortAndCopyBin( iBin - 1, allXHits );
          } else {
            assert( !allXHits.candidateStartIndex.empty() );
            // merge candidate index range with previous candidate
            allXHits.candidateStartIndex.pop_back();
            allXHits.candidateEndIndex.pop_back();
          }

          if ( iBin != lastBinCopied && iBin != last2ndBinCopied ) { sortAndCopyBin( iBin, allXHits ); }

          sortAndCopyBin( iBin + 1, allXHits );

        } else if ( iBin != lastBinCopied ) {
          sortAndCopyBin( iBin, allXHits );
        } else {
          // we already got this bin included in the previous range
          allXHits.candidateStartIndex.pop_back();
          allXHits.candidateEndIndex.pop_back();
        }
        allXHits.candidateEndIndex.push_back( allXHits.size() );
        lastBinCopied    = iBin + checkNeighbours;
        last2ndBinCopied = iBin;
      }
    }

    /**
     * @brief Scanning through encoded planes in each bin to select promising candidates.
     * @param minPlanes Threshold number of different planes in bin.
     * @param minPlanesComplX Threshold number of different x planes including complementary x planes from neighbour
     * bins.
     * @param minPlanesComplUV Threshold number of different stereo planes including complementary stereo planes from
     * neighbour bins.
     * @param minPlanesCompl Threshold number of total different planes including complementary planes from neighbour
     * bins.
     * @param minKeepBoth Minimum number of planes in single bin leading to it being selected in any case.
     * @return Number of candidates.
     * @details Most hits from true tracks a distributed across 1,2 or 3 bins. By simply selecting bins over threshold,
     * a lot of clones are picked up if several of the contributing bins are above threshold. Therefore in the remove
     * step, the next bin in the list of candidates after threshold scan is peeked and if it's the direct neighbour only
     * the one with more total planes is kept.
     */
    auto HoughTransformation::pickUpCandidateBins( int minPlanes, int minPlanesComplX, int minPlanesComplUV,
                                                   int minPlanesCompl ) {
      direct_debug( "------------pickUpCandidateBins------------" );
      direct_debug( "" );
      // threshold values imply "equal or less" which is not implemented in SIMDWrapper thus simply decrement
      --minPlanes;
      // it's better to first quickly collect all bins that have at least a minimum amount of planes
      for ( auto iBin{ minBinOffset }; iBin < nBins; iBin += simd::size ) {
        const auto aboveThreshold = decodeNbDifferentPlanes<simd::int_v>( iBin ) > minPlanes;
        compressstoreCandidates( aboveThreshold, simd::indices( iBin ) );
      }
      direct_debug( m_candSize, "over first threshold nDifferentPlanes > minPlanes =", minPlanes );
      const auto originalSize = m_candSize;
      m_candSize              = 0;
      // then check if there's really a local excess of planes
      // this vectorised version is marginally faster than a simple remove if
      for ( auto i{ 0 }; i < originalSize; i += simd::size ) {
        const auto loopmask = simd::loop_mask( i, originalSize );
        const auto iBin     = select( loopmask, simd::int_v{ m_candidateBins.data() + i }, minBinOffset );
        assert( none( iBin < minBinOffset ) );
        const auto pc      = gather( m_planeCounter.data(), iBin ) >> nDiffPlanesBits;
        const auto pcLeft  = gather( m_planeCounter.data(), iBin - 1 ) >> nDiffPlanesBits;
        const auto pcRight = gather( m_planeCounter.data(), iBin + 1 ) >> nDiffPlanesBits;

        const auto pcTot  = pc | pcLeft | pcRight;
        const auto nTot   = popcount_v( pcTot );
        const auto nTotX  = popcount_v( pcTot & xFlags );
        const auto nTotUV = popcount_v( pcTot & uvFlags );

        const auto removeMask = nTot < minPlanesCompl || nTotX < minPlanesComplX || nTotUV < minPlanesComplUV;
        compressstoreCandidates( !removeMask && loopmask, iBin );
      }
      direct_debug( m_candSize, "over neighbour thresholds nTot >= (minPlanesCompl =", minPlanesCompl,
                    "&& minPlanesComplX =", minPlanesComplX, "&& minPlanesComplUV =", minPlanesComplUV, ")" );
      /**
       * the problem now is that we picked some bins "twice", meaning that we have two adjacent bins
       * which are really only one candidate and were only selected both, because we are checking the neighbours for
       * planes. This remove_if mitigates this problem. It's a trade off between efficiencies and speed because
       * we might loose some hits (and even candidates) by removing one the bins. Always removing the one that has less
       * total planes and if both have an equal number of total planes removing the next (right) one has a good tradeoff
       * between speed and efficiencies. Always keeping both sacrifices ~5% throughput for O(0.1%) efficiency.
       */
      auto candSpan = LHCb::make_span( m_candidateBins.begin(), m_candSize );
      auto removeNext{ false };
      m_candSize =
          std::distance( candSpan.begin(), std::remove_if( candSpan.begin(), candSpan.end(), [&]( const auto& iBin ) {
                           if ( removeNext ) {
                             removeNext = false;
                             return true;
                           }
                           const auto iBinNext = *( std::next( &iBin ) );
                           const auto adjacent = iBin == iBinNext - 1;
                           if ( !adjacent || iBin == candSpan.back() ) return false;

                           const auto pcLeft   = planeBits<scalar::int_v>( iBin - 1 );
                           const auto pc       = planeBits<scalar::int_v>( iBin );
                           const auto pcRight  = planeBits<scalar::int_v>( iBin + 1 );
                           const auto pcRight2 = planeBits<scalar::int_v>( iBin + 2 );

                           const auto nTot     = popcount_v( pc | pcRight | pcLeft ).cast();
                           const auto nTotNext = popcount_v( pc | pcRight | pcRight2 ).cast();

                           removeNext = nTot >= nTotNext;
                           return !removeNext;
                         } ) );
      return m_candSize;
    }

  } // namespace

  template <typename TrackType>
  class PrForwardTracking
      : public LHCb::Algorithm::Transformer<TracksFT( const FT::Hits&, const TrackType&, const ZoneCache&,
                                                      const IPrAddUTHitsTool&, const DeMagnet& ),
                                            LHCb::Algorithm::Traits::usesConditions<ZoneCache, DeMagnet>> {
  public:
    using base_class_t = LHCb::Algorithm::Transformer<TracksFT( const FT::Hits&, const TrackType&, const ZoneCache&,
                                                                const IPrAddUTHitsTool&, const DeMagnet& ),
                                                      LHCb::Algorithm::Traits::usesConditions<ZoneCache, DeMagnet>>;
    using base_class_t::addConditionDerivation;
    using base_class_t::debug;
    using base_class_t::error;
    using base_class_t::info;
    using base_class_t::inputLocation;
    using base_class_t::msgLevel;
    using ErrorCounter = Gaudi::Accumulators::MsgCounter<MSG::ERROR>;

    /********************************************************************************
     * Standard Constructor, initialise member variables (such as NN variables) here.
     * Paths are figured out by the scheduler.
     ********************************************************************************/
    PrForwardTracking( std::string const& name, ISvcLocator* pSvcLocator )
        : base_class_t( name, pSvcLocator,
                        { typename base_class_t::KeyValue{ "SciFiHits", "" },
                          typename base_class_t::KeyValue{ "InputTracks", "" },
                          typename base_class_t::KeyValue{ "FTZoneCache", std::string{ ZoneCache::Location } + name },
                          typename base_class_t::KeyValue{ "AddUTHitsToolName", "PrAddUTHitsTool" },
                          typename base_class_t::KeyValue{ "Magnet", LHCb::Det::Magnet::det_path } },
                        typename base_class_t::KeyValue{ "OutputTracks", "" } )
        , m_NN{ NNVars }
        , m_NNVeloUT{ NNVarsVeloUT } {}

    /********************************************************************************
     * Initialisation, set some member variables and cache
     ********************************************************************************/
    StatusCode initialize() override {
      auto sc = base_class_t::initialize();
      if ( sc.isFailure() ) return sc;
      this->template addConditionDerivation<ZoneCache( const DeFT& )>( { DeFTDetectorLocation::Default },
                                                                       this->template inputLocation<ZoneCache>() );
      if ( m_timerTool.isEnabled() ) m_timerIndex = m_timerTool->addTimer( this->name() );
      return sc;
    }

    // main call
    TracksFT operator()( const FT::Hits&, const TrackType&, const ZoneCache&, const IPrAddUTHitsTool&,
                         const DeMagnet& ) const override;
    auto     forwardTracks( const TrackType&, const FT::Hits&, const ZoneCache&, const DeMagnet& ) const;

    auto calculateMomentumBorders( const VeloSeed& ) const;

    template <bool secondLoop>
    void selectXCandidates( PrForwardTracks&, const VeloSeedExtended&, ModSciFiHits::ModPrSciFiHitsSOA&,
                            const FT::Hits&, const PrParametersX& ) const;

    template <bool STORE_DATA = false>
    auto calcMVAInput( const VeloSeedExtended&, PrForwardTrack&, float ) const;

    template <bool secondLoop, typename Buffer>
    auto selectFullCandidates( LHCb::span<PrForwardTrack>, PrParametersX, const ZoneCache&, const VeloSeedExtended&,
                               const FT::Hits&, float, Buffer& ) const;

    auto removeDuplicates( PrForwardTracks&, const FT::Hits& ) const;

    template <typename Container>
    auto makeLHCbLongTracks( PrForwardTracks&&, Container&&, const TrackType&, const IPrAddUTHitsTool& ) const;

    auto make_TracksFT_from_ancestors( const TrackType& input_tracks ) const {
      const TracksVP*                  velo_ancestors{ nullptr };
      const TracksUT*                  upstream_ancestors{ nullptr };
      const LHCb::Pr::Seeding::Tracks* seed_ancestors{ nullptr };
      const auto                       history = Event::Enum::Track::History::PrForward;
      if constexpr ( std::is_same_v<TrackType, TracksUT> ) {
        velo_ancestors     = input_tracks.getVeloAncestors();
        upstream_ancestors = &input_tracks;
      } else {
        velo_ancestors = &input_tracks;
      }
      return TracksFT{ velo_ancestors,
                       upstream_ancestors,
                       seed_ancestors,
                       history,
                       Zipping::generateZipIdentifier(),
                       { input_tracks.get_allocator().resource() } };
    }

  private:
    mutable Gaudi::Accumulators::SummingCounter<> m_inputTracksCnt{ this, "Input tracks" };
    mutable Gaudi::Accumulators::SummingCounter<> m_acceptedInputTracksCnt{ this, "Accepted input tracks" };
    mutable Gaudi::Accumulators::SummingCounter<> m_outputTracksCnt{ this, "Created long tracks" };
    mutable Gaudi::Accumulators::SummingCounter<> m_duplicateCnt{ this, "Removed duplicates" };
    mutable Gaudi::Accumulators::SummingCounter<> m_secondLoopCnt{ this, "Percentage second loop execution" };
    mutable Gaudi::Accumulators::SigmaCounter<> m_xCandidates1Cnt{ this, "Number of x candidates per track 1st Loop" };
    mutable Gaudi::Accumulators::SigmaCounter<> m_xCandidates2Cnt{ this, "Number of x candidates per track 2nd Loop" };
    mutable Gaudi::Accumulators::StatCounter<>  m_candidateBinsCnt{ this, "Number of candidate bins per track" };
    mutable Gaudi::Accumulators::StatCounter<> m_candidates1Cnt{ this, "Number of complete candidates/track 1st Loop" };
    mutable Gaudi::Accumulators::StatCounter<> m_candidates2Cnt{ this, "Number of complete candidates/track 2nd Loop" };
    mutable Gaudi::Accumulators::Counter<>     m_noSciFiHits{ this, "Empty SciFi hits" };
    mutable Gaudi::Accumulators::Counter<>     m_noInputTracks{ this, "Empty input tracks" };
    mutable Gaudi::Accumulators::Counter<>     m_noHitsAndTracks{ this, "Empty input tracks AND SciFi hits" };

    // avoid sign comparison warning
    Gaudi::Property<unsigned> m_minTotalHits{ this, "MinTotalHits", 10 };
    Gaudi::Property<bool>     m_secondLoop{ this, "SecondLoop", true };

    Gaudi::Property<float> m_minPT{ this, "MinPT", 50.f * Gaudi::Units::MeV };
    Gaudi::Property<float> m_minP{ this, "MinP", 1500.f * Gaudi::Units::MeV };
    Gaudi::Property<int>   m_minPlanesCompl{ this, "MinPlanesCompl", 10 };
    Gaudi::Property<int>   m_minPlanes{ this, "MinPlanes", 4 };
    Gaudi::Property<int>   m_minPlanesComplX{ this, "MinPlanesComplX", 4 };
    Gaudi::Property<int>   m_minPlanesComplUV{ this, "MinPlanesComplUV", 4 };
    Gaudi::Property<int>   m_maxLinearYEndT{ this, "MaxLinearYEndT", 2600 * Gaudi::Units::mm };

    // first loop Hough Cluster search
    // unsigned to avoid warnings
    Gaudi::Property<unsigned> m_minXHits{ this, "MinXHits", 5 };
    Gaudi::Property<float>    m_maxXWindow{ this, "MaxXWindow", 1.2f * Gaudi::Units::mm };
    Gaudi::Property<float>    m_maxXWindowSlope{ this, "MaxXWindowSlope", 0.002f / Gaudi::Units::mm };
    Gaudi::Property<float>    m_maxXGap{ this, "MaxXGap", 1.2f * Gaudi::Units::mm };
    Gaudi::Property<int>      m_minSingleHits{ this, "MinSingleHits", 2 };
    Gaudi::Property<float>    m_maxXSubrangeWidth{ this, "MaxXSubrangeWidth", 1.f * Gaudi::Units::mm };

    // second loop Hough Cluster search
    // unsigned to avoid warnings
    Gaudi::Property<unsigned> m_minXHits2nd{ this, "MinXHits2nd", 4 };
    Gaudi::Property<float>    m_maxXWindow2nd{ this, "MaxXWindow2nd", 1.5f * Gaudi::Units::mm };
    Gaudi::Property<float>    m_maxXWindowSlope2nd{ this, "MaxXWindowSlope2nd", 0.002f / Gaudi::Units::mm };
    Gaudi::Property<float>    m_maxXGap2nd{ this, "MaxXGap2nd", 0.5f * Gaudi::Units::mm };

    Gaudi::Property<float> m_maxChi2LinearFit{ this, "MaxChi2LinearFit", 100.f };
    Gaudi::Property<float> m_maxChi2XProjection{ this, "MaxChi2XProjection", 15.f };
    Gaudi::Property<float> m_maxChi2PerDoF{ this, "MaxChi2PerDoF", 7.f };
    Gaudi::Property<float> m_maxChi2PerDoFFinal{ this, "MaxChi2PerDoFFinal", 4.f };
    Gaudi::Property<float> m_maxChi2XAddLinear{ this, "MaxChi2XAddLinear", 2000.f };
    Gaudi::Property<float> m_maxChi2XAddFull{ this, "MaxChi2XAddFull", 500.f };

    Gaudi::Property<float> m_tolYUVSearch{ this, "TolYUVSearch", 11.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_tolY{ this, "TolY", 5.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_tolYSlope{ this, "TolYSlope", 0.002f / Gaudi::Units::mm };
    Gaudi::Property<float> m_maxTolY{ this, "MaxTolY", 2000.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_uvSearchBinWidth{ this, "UVSearchBinWidth", 2.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_tolYMag{ this, "TolYMag", 10.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_tolYMagSlope{ this, "TolYMagSlope", 0.015f / Gaudi::Units::mm };
    Gaudi::Property<float> m_maxChi2StereoLinear{ this, "MaxChi2StereoLinear", 60.f };
    Gaudi::Property<float> m_maxChi2Stereo{ this, "MaxChi2Stereo", 4.5f };
    Gaudi::Property<float> m_maxChi2StereoAdd{ this, "MaxChi2StereoAdd", 4.5f };

    // unsigned to avoid warnings
    Gaudi::Property<unsigned> m_minStereoHits{ this, "MinStereoHits", 4 };
    Gaudi::Property<float>    m_tolYTriangleSearch{ this, "TolYTriangleSearch",
                                                 Detector::FT::triangleHeight* Gaudi::Units::mm };

    // Momentum guided search window switch
    Gaudi::Property<bool>  m_useMomentumSearchWindow{ this, "UseMomentumSearchWindow", false };
    Gaudi::Property<bool>  m_useWrongSignWindow{ this, "UseWrongSignWindow", true };
    Gaudi::Property<float> m_wrongSignPT{ this, "WrongSignPT", 2000.f * Gaudi::Units::MeV };

    // calculate an upper error boundary on the tracks x-prediction,
    Gaudi::Property<float> m_upperLimitOffset{ this, "UpperLimitOffset", 100.f };
    Gaudi::Property<float> m_upperLimitSlope{ this, "UpperLimitSlope", 2800.f };
    Gaudi::Property<float> m_upperLimitMax{ this, "UpperLimitMax", 600.f };
    Gaudi::Property<float> m_upperLimitMin{ this, "UpperLimitMin", 150.f };
    // same as above for the lower limit
    Gaudi::Property<float> m_lowerLimitOffset{ this, "LowerLimitOffset", 50.f };
    Gaudi::Property<float> m_lowerLimitSlope{ this, "LowerLimitSlope", 1400.f };
    Gaudi::Property<float> m_lowerLimitMax{ this, "LowerLimitMax", 600.f };

    // Track Quality (Neural Net)
    Gaudi::Property<float> m_maxDistX{ this, "MaxDistX", 140.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_maxDistY{ this, "MaxDistY", 500.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_maxY0Diff{ this, "MaxY0Diff", 140.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_maxYSlopeDiff{ this, "MaxYSlopeDiff", 0.055f };
    Gaudi::Property<float> m_minQuality{ this, "MinQuality", 0.14f };
    Gaudi::Property<float> m_minQualityMomentum{ this, "MinQualityMomentum", 0.1f };
    Gaudi::Property<float> m_deltaQuality{ this, "DeltaQuality", 0.24f };

    // duplicate removal
    Gaudi::Property<float> m_minXInterspace{ this, "MinXInterspace", 50.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_minYInterspace{ this, "MinYInterspace", 100.f * Gaudi::Units::mm };
    Gaudi::Property<float> m_maxCommonFrac{ this, "MaxCommonFrac", 0.5f };

    ReadGhostNN       m_NN;
    ReadGhostNNVeloUT m_NNVeloUT;

    ToolHandle<IPrDebugTrackingTool>              m_debugTool{ this, "DebugTool", "" };
    mutable PublicToolHandle<ISequencerTimerTool> m_timerTool{ this, "TimerTool", "",
                                                               "Do not use in combination with multi-threading." };
    int                                           m_timerIndex{};
  };

  DECLARE_COMPONENT_WITH_ID( PrForwardTracking<TracksUT>, "PrForwardTracking" )
  DECLARE_COMPONENT_WITH_ID( PrForwardTracking<TracksVP>, "PrForwardTrackingVelo" )

  /**
   * @brief Main execution of the Forward Tracking.
   * @param SciFiHits SOAContainer containing SciFi hits.
   * @param input_tracks Tracks that shall be extended to SciFi stations. Either Velo tracks or Upstream tracks.
   * @param cache Cache containing useful information about SciFi zones.
   */
  template <typename TrackType>
  TracksFT PrForwardTracking<TrackType>::operator()( const FT::Hits& SciFiHits, const TrackType& input_tracks,
                                                     const ZoneCache& cache, const IPrAddUTHitsTool& addUTHits,
                                                     const DeMagnet& magnet ) const {
    const auto scopedTimer = m_timerTool.get()->scopedTimer( m_timerIndex, m_timerTool.isEnabled() );
    // check if we can do any Long tracking, return empty container if not
    if ( SciFiHits.empty() || input_tracks.empty() ) {
      if ( SciFiHits.empty() ) { ++m_noSciFiHits; }
      if ( input_tracks.empty() ) { ++m_noInputTracks; }
      if ( SciFiHits.empty() && input_tracks.empty() ) { ++m_noHitsAndTracks; }
      return make_TracksFT_from_ancestors( input_tracks );
    }

    assert( [&] {
      auto allSentinels = true;
      for ( unsigned int iZone{ 0 }; iZone < Detector::FT::nZonesTotal; ++iZone ) {
        const auto zoneEnd = SciFiHits.getZoneIndices( iZone ).second;
        allSentinels &= ( SciFiHits.x( zoneEnd ) > 2 * HoughTransformation::rangeMax );
      }
      return allSentinels;
    }() && "End of zones in PrSciFiHits are not protected by large x values as expected by the Forward Tracking." );

    auto candidates = forwardTracks( input_tracks, SciFiHits, cache, magnet );

    const auto idSets = removeDuplicates( candidates, SciFiHits );

    return makeLHCbLongTracks( std::move( candidates ), std::move( idSets ), input_tracks, addUTHits );
  }

  /**
   * @brief Method calling the main components of the Forward Tracking, i.e. "forwarding the tracks".
   * @param input_tracks Tracks that shall be extended to SciFi stations. Either Velo tracks or Upstream tracks.
   * @param SciFiHits SOAContainer containing SciFi hits.
   * @param cache Cache containing useful information about SciFi zones.
   */
  template <typename TrackType>
  auto PrForwardTracking<TrackType>::forwardTracks( const TrackType& input_tracks, const FT::Hits& SciFiHits,
                                                    const ZoneCache& cache, const DeMagnet& magnet ) const {

    m_inputTracksCnt += input_tracks.size();
    auto xCandidates1Cnt  = m_xCandidates1Cnt.buffer();
    auto xCandidates2Cnt  = m_xCandidates2Cnt.buffer();
    auto candidateBinsCnt = m_candidateBinsCnt.buffer();
    auto candidates1Cnt   = m_candidates1Cnt.buffer();
    auto candidates2Cnt   = m_candidates2Cnt.buffer();
    auto secondLoopCnt    = m_secondLoopCnt.buffer();

    PrForwardTracks candidates{};
    PrForwardTracks tracks{};
    candidates.reserve( input_tracks.size() );
    tracks.reserve( input_tracks.size() );

    // The HoughTransformation object is huge so let's put it on the heap
    auto  houghTransformation_ptr = std::make_unique<HoughTransformation>();
    auto& houghTransformation     = *houghTransformation_ptr.get();

    ModSciFiHits::ModPrSciFiHitsSOA allXHits{ Zipping::generateZipIdentifier(),
                                              { input_tracks.get_allocator().resource() } };
    allXHits.init( SciFiHits.size(), HoughTransformation::nBins );

    PrParametersX pars{ m_minXHits,      m_maxXWindow,    m_maxXWindowSlope,    m_maxXGap,
                        m_minStereoHits, m_maxChi2PerDoF, m_maxChi2XProjection, m_maxChi2LinearFit };
    PrParametersX pars2ndLoop{ m_minXHits2nd,   m_maxXWindow2nd, m_maxXWindowSlope2nd, m_maxXGap2nd,
                               m_minStereoHits, m_maxChi2PerDoF, m_maxChi2XProjection, m_maxChi2LinearFit };

    const auto magScaleFactor = static_cast<float>( magnet.signedRelativeCurrent() );
    direct_debug( cache );
    direct_debug( "magScaleFactor =", magScaleFactor );
    // main loop over input tracks
    const auto inputTrks = input_tracks.scalar();
    auto       acceptedInputTracks{ 0 };
    VeloSeed   previousSeed{};
    for ( const auto& tr : inputTrks ) {
      const auto iTrack = tr.offset();
      direct_debug( "" );
      direct_debug( "============================VELO TRACK No.", iTrack, "=================================" );
      direct_debug( "" );

      const auto [endv_pos, endv_dir, qOverP] = [&] {
        if constexpr ( std::is_same_v<TrackType, TracksUT> ) {
          const auto velozipped = input_tracks.getVeloAncestors()->scalar();
          const auto trackVP    = tr.trackVP();
          const auto velo_track = velozipped[trackVP.cast()];
          const auto endv_pos   = velo_track.StatePos( Event::Enum::State::Location::EndVelo );
          const auto endv_dir   = velo_track.StateDir( Event::Enum::State::Location::EndVelo );
          return std::tuple{ endv_pos, endv_dir, tr.qOverP().cast() };
        } else {
          const auto endv_pos = tr.StatePos( Event::Enum::State::Location::EndVelo );
          const auto endv_dir = tr.StateDir( Event::Enum::State::Location::EndVelo );
          return std::tuple{ endv_pos, endv_dir, nanMomentum };
        }
      }();

      // remove tracks that are so steep in y that they go out of acceptance
      if ( const auto yStraightEndT =
               endv_pos.y() +
               abs( endv_dir.y() ) * ( static_cast<float>( Z( State::Location::EndT ).value() ) - endv_pos.z() );
           yStraightEndT.cast() > m_maxLinearYEndT ) {
        direct_debug( "yStraightEndT =", yStraightEndT, "not in acceptance  ... skip" );
        continue;
      }

      const VeloSeed seed{ endv_pos.x().cast(), endv_pos.y().cast(), endv_pos.z().cast(),
                           endv_dir.x().cast(), endv_dir.y().cast(), qOverP,
                           magScaleFactor };

      direct_debug( seed );
      // there can be clones coming from the Velo tracking
      if ( seed.essentiallyEqual( previousSeed ) ) {
        direct_debug( "Same as previous seed", previousSeed, "...skip" );
        continue;
      }
      previousSeed = seed;

      if constexpr ( std::is_same_v<TrackType, TracksUT> ) {
        if ( !std::isnan( qOverP ) ) {
          // we have VeloUT to make a preselection according to momentum
          const auto p  = std::abs( 1.f / qOverP ) * float{ Gaudi::Units::MeV };
          const auto pt = p * seed.momProj * float{ Gaudi::Units::MeV };
          if ( p < m_minP || pt < m_minPT ) continue;
        }
      }

      ++acceptedInputTracks;

      const VeloSeedExtended veloSeed{ iTrack, seed, cache };

      houghTransformation.clear();

      const auto momentumBorders = calculateMomentumBorders( seed );
      direct_debug( "momentumBorders @ S3L0 = [", momentumBorders.first, ",", momentumBorders.second, "]" );
      houghTransformation.projectStereoHitsToHoughSpace( veloSeed, cache, momentumBorders, SciFiHits );
      houghTransformation.projectXHitsToHoughSpace( veloSeed, momentumBorders, SciFiHits );

      const auto nCandBins = houghTransformation.pickUpCandidateBins( m_minPlanes, m_minPlanesComplX,
                                                                      m_minPlanesComplUV, m_minPlanesCompl );
      direct_debug( houghTransformation );

      candidateBinsCnt += nCandBins;
      if ( !nCandBins ) continue;

      allXHits.clear();
      houghTransformation.sortAndCopyBinContents( m_minTotalHits, allXHits );

      assert( [&] {
        const auto coords  = LHCb::make_span( allXHits.data<ModSciFiHits::HitTag::coord>(), allXHits.size() );
        const auto end     = coords.end();
        auto       coordIt = std::is_sorted_until( coords.begin(), end );
        while ( coordIt != end ) {
          const auto c2 = *coordIt;
          const auto c1 = *std::prev( coordIt );
          // Because of the way these coordinates are sorted it can happen that an almost equal value
          // ends up being unsorted (see HoughTransformation::calculateBinNumber). I'll allow this as
          // long the difference is negligible w.r.t to the general significance of the values, i.e.
          // 100 microns.
          constexpr auto maxRelevantDiff = 100.0f * Gaudi::Units::um;
          if ( std::abs( c2 - c1 ) < maxRelevantDiff ) {
            coordIt = std::is_sorted_until( coordIt, end );
          } else {
            return false;
          }
        }
        return true;
      }() );

      candidates.clear();
      selectXCandidates<false>( candidates, veloSeed, allXHits, SciFiHits, pars );
      const auto candStart2nd = candidates.size();
      xCandidates1Cnt += candStart2nd;
      auto [good, ok] =
          selectFullCandidates<false>( candidates, pars, cache, veloSeed, SciFiHits, magScaleFactor, candidates1Cnt );
      secondLoopCnt += !good;
      if ( !good && m_secondLoop ) {
        direct_debug( "" );
        direct_debug( "Now second loop" );
        direct_debug( "" );
        const auto candStart2nd = candidates.size();
        selectXCandidates<true>( candidates, veloSeed, allXHits, SciFiHits, pars2ndLoop );
        xCandidates2Cnt += candidates.size();
        auto       candidates2ndLoop = LHCb::make_span( candidates.begin() + candStart2nd, candidates.end() );
        const auto ok2nd = selectFullCandidates<true>( candidates2ndLoop, pars2ndLoop, cache, veloSeed, SciFiHits,
                                                       magScaleFactor, candidates2Cnt )
                               .second;
        // a short circuit booleon eval would be wrong here!
        ok = ok ? ok : ok2nd;
      }
      if ( ok ) {
        const auto bestQuality =
            std::max_element( candidates.begin(), candidates.end(), []( const auto& t1, const auto& t2 ) {
              return t1.quality() < t2.quality();
            } )->quality();
        const auto selectQuality = bestQuality - m_deltaQuality;
        direct_debug( "Make selection on quality relative to best quality", bestQuality, ", selectQuality is",
                      selectQuality );
        std::copy_if( std::make_move_iterator( candidates.begin() ), std::make_move_iterator( candidates.end() ),
                      std::back_inserter( tracks ), [&]( const auto& track ) {
                        direct_debug( track );
                        return track.quality() >= selectQuality && track.valid();
                      } );
      }
    } // ================end of input track loop====================
    m_acceptedInputTracksCnt += acceptedInputTracks;
    return tracks;
  }

  /**
   * @brief select candidate tracks using x hits
   * @param candidates container for x candidates
   * @param veloSeed useful info from the input track
   * @param allXHits contains all x hits that might form a candidate
   * @param SciFiHits PrSciFiHits container
   * @param xPars x parameter bundle
   * @details Loops over previously found ranges of the x hits trying to form one or more track candidates out of them.
   * Candidates must have a minimum number of planes, a central part of the x positions on the reference plane have to
   * lie within a window. To get clean candidates, at least two fits of the x projection are performed. In case a
   * candidate was found the used hits are removed from the range such that they are not picked up again while iterating
   * further through the range or in a second loop.
   */
  template <typename TrackType>
  template <bool secondLoop>
  void PrForwardTracking<TrackType>::selectXCandidates( PrForwardTracks& candidates, const VeloSeedExtended& veloSeed,
                                                        ModSciFiHits::ModPrSciFiHitsSOA& allXHits,
                                                        const FT::Hits& SciFiHits, const PrParametersX& pars ) const {
    if constexpr ( secondLoop )
      direct_debug( "------------selectXCandidates------------secondLoop" );
    else
      direct_debug( "------------selectXCandidates------------" );
    direct_debug( "" );
    direct_debug( veloSeed );
    XCandidate            protoCand{ allXHits, SciFiHits };
    static_vector<int, 6> otherPlanes{};

    const auto startIndexSize = allXHits.candidateStartIndex.size();
    assert( allXHits.candidateStartIndex.size() == allXHits.candidateEndIndex.size() );
    // unsigned to avoid warning
    for ( unsigned iStart{ 0 }; iStart < startIndexSize; ++iStart ) {
      direct_debug( "++++ XCandidate", iStart, "++++" );
      auto idx1   = allXHits.candidateStartIndex[iStart];
      auto idxEnd = allXHits.candidateEndIndex[iStart];
      for ( int idx2 = idx1 + pars.minXHits; idx2 <= idxEnd; idx2 = idx1 + pars.minXHits ) {
        // like all good containers it's [idx1,idx2)
        const float xWindow = pars.maxXWindow + ( std::abs( allXHits.coord( idx1 ) ) +
                                                  std::abs( allXHits.coord( idx1 ) - veloSeed.xStraightAtRef ) ) *
                                                    pars.maxXWindowSlope;
        direct_debug( "xWindow from params =", xWindow, ">",
                      "xAtRef cluster width =", ( allXHits.coord( idx2 - 1 ) - allXHits.coord( idx1 ) ) );
        // hits must fit into a window
        if ( ( allXHits.coord( idx2 - 1 ) - allXHits.coord( idx1 ) ) > xWindow ) {
          ++idx1;
          continue;
        }

        protoCand.clear();
        for ( auto idx{ idx1 }; idx < idx2; ++idx ) { protoCand.addHit( idx ); }
        // try to find more hits to the right
        idx2 = protoCand.improveRightSide( idx1, idx2, idxEnd, pars.maxXGap, xWindow );

        if ( protoCand.nDifferentPlanes() < pars.minXHits ) {
          ++idx1;
          continue;
        }

        const auto nSinglePlanes = protoCand.nSinglePlanes();
        const auto multiple      = nSinglePlanes != protoCand.nDifferentPlanes();
        if ( nSinglePlanes >= m_minSingleHits && multiple ) {
          separateSingleHitsForFit( otherPlanes, protoCand, veloSeed );
          protoCand.solveLineFit();
          // add best other hit on empty plane, after this there is only one hit per plane
          addBestOtherHits( otherPlanes, protoCand, veloSeed );
        } else if ( !multiple && !secondLoop ) {
          // in the first loop wrong hits at the edges of the found "cluster" can be removed
          // efficiently by trying to find a very small subrange
          protoCand.tryToShrinkRange( idx1, idx2, m_maxXSubrangeWidth, m_minPlanesComplX );
        } else {
          // if every plane has at least two hits these are sorted out by the linear fit
          prepareAllHitsForFit( allXHits, protoCand );
        }
        // at this point there can also be less than minXHits (coming from shrinking of range)
        initXFitParameters( protoCand, veloSeed );
        direct_debug( "xParams after initialisation =", protoCand.getXParams() );

        fitLinearXProjection<secondLoop>( protoCand, pars, SciFiHits, veloSeed );
        if ( fillEmptyXPlanes<secondLoop>( protoCand, pars, veloSeed, SciFiHits, m_maxChi2XAddLinear ) ) {
          fitLinearXProjection<secondLoop>( protoCand, pars, SciFiHits, veloSeed );
        }
        direct_debug( "xParams after linear fit     =", protoCand.getXParams() );
        auto ok = protoCand.nDifferentPlanes() >= pars.minXHits;
        if ( ok ) {
          initYFitParameters( protoCand, veloSeed );
          ok = fitXProjection( protoCand, pars, SciFiHits );
          if ( ok ) {
            if ( fillEmptyXPlanes<secondLoop>( protoCand, pars, veloSeed, SciFiHits, m_maxChi2XAddFull ) ) {
              ok = fitXProjection( protoCand, pars, SciFiHits );
            }
            direct_debug( "ok =", ok, protoCand );
          }
          if ( ok ) {
            // It's a track!
            ok                                 = removeUsedHits( idx1, idxEnd, protoCand, allXHits );
            allXHits.candidateEndIndex[iStart] = idxEnd;
            // TODO: in PrVeloUTFilter mode, an LDA could also use momentum information and be a powerful early ghost
            // reduction. Using the debug tool all necessary variables can easily be obtained here.
            candidates.emplace_back( protoCand.getCoordsToFit(), protoCand.getXParams(), protoCand.getYParams(),
                                     protoCand.getChi2NDoF(), veloSeed.iTrack );
          } else {
            // set idx1 to the most left hit on the reference plane from fitted hits
            // this avoids constructing the same candidate again from a overlapping set of hits
            // few low momentum tracks are lost, but the throughput improvement is worth it
            const auto& c = protoCand.getCoordsToFit();
            while ( idx1 < idxEnd && std::find( c.begin(), c.end(), allXHits.fulldex( idx1 ) ) == c.end() ) { ++idx1; }
          }
        }
        idx1 += !ok;
      }
      direct_debug( "" );
    }
  }

  /**
   * @brief creates the input for the ghost killing MVA
   * @param qOverP charge over momentum estimate for the track
   * @param veloSeed velo track extension
   * @param track track candidate
   * @return array of MVA input values
   */
  template <typename TrackType>
  template <bool STORE_DATA>
  auto PrForwardTracking<TrackType>::calcMVAInput( const VeloSeedExtended& veloSeed, PrForwardTrack& track,
                                                   float magScaleFactor ) const {

    const auto& yPars       = track.getYParams();
    const auto& xPars       = track.getXParams();
    const auto  tx          = veloSeed.seed.tx;
    const auto  ty          = veloSeed.seed.ty;
    const auto  dSlope      = xPars[1] - tx;
    const auto  ayPredicted = veloSeed.yStraightAtRef + veloSeed.calcYCorr( dSlope );
    const auto  byPredicted = ty + veloSeed.calcTyCorr( dSlope );

    const auto xAtRef      = xPars[0];
    const auto zMagPrecise = veloSeed.calcZMag( dSlope );
    const auto xMag        = veloSeed.seed.x( zMagPrecise );
    const auto bxPredicted = ( xAtRef - xMag ) / ( zReference - zMagPrecise );
    const auto dYParam0    = std::abs( ayPredicted - yPars[0] );
    const auto dYParam1    = std::abs( byPredicted - yPars[1] );
    const auto dXParam1    = std::abs( bxPredicted - xPars[1] );

    const auto qOverP  = track.estimateChargeOverMomentum( veloSeed, magScaleFactor );
    const auto xMatch  = calcXMatchDistance( track, veloSeed );
    const auto yMatch  = calcYMatchDistance( track, veloSeed );
    const auto redChi2 = track.getChi2PerNDoF();

    if constexpr ( std::is_same_v<TrackType, TracksUT> && !STORE_DATA ) {
      return std::array{ vdt::fast_logf( std::abs( 1.f / qOverP - 1.f / veloSeed.seed.qOverP ) ),
                         redChi2,
                         xMatch,
                         std::abs( yMatch ),
                         dYParam0,
                         dYParam1,
                         std::abs( ty ),
                         std::abs( qOverP ),
                         std::abs( tx ),
                         dXParam1 };
    } else if constexpr ( !STORE_DATA ) {
      return std::array{ redChi2,  xMatch,         std::abs( yMatch ), dYParam0,
                         dYParam1, std::abs( ty ), std::abs( qOverP ), std::abs( tx ),
                         dXParam1 };
    } else {
      const auto dz     = static_cast<float>( StateParameters::ZEndT ) - zReference;
      const auto xEndT  = track.x( dz );
      const auto yEndT  = track.y( dz );
      const auto txEndT = track.xSlope( dz );
      const auto tyEndT = track.ySlope( dz );
      // unfortunately we are restricted to a vector here (span does not work in the interface)
      std::vector<int> indices;
      indices.reserve( track.size() );
      std::copy( track.getCoordsToFit().begin(), track.getCoordsToFit().end(), std::back_inserter( indices ) );
      std::array<IPrDebugTrackingTool::VariableDef, 27> vars_and_values = { {
          { "label", m_debugTool->check( veloSeed.iTrack, 0, indices ) },
          { "x", veloSeed.seed.x0 },
          { "y", veloSeed.seed.y0 },
          { "tx", veloSeed.seed.tx },
          { "ty", veloSeed.seed.ty },
          { "qopUT", veloSeed.seed.qOverP },
          { "redChi2", track.getChi2PerNDoF() },
          { "zMagMatch", veloSeed.calcZMag( dSlope ) },
          { "zMag", veloSeed.zMag },
          { "dSlope", dSlope },
          { "xEndT", xEndT },
          { "yEndT", yEndT },
          { "txEndT", txEndT },
          { "tyEndT", tyEndT },
          { "qop", qOverP },
          { "ySeedMatch", calcYMatchDistance( track, veloSeed ) + yEndT },
          { "yParam0Final", yPars[0] },
          { "yParam1Final", yPars[1] },
          { "yParam2Final", yPars[2] },
          { "xParam0Final", xPars[0] },
          { "xParam1Final", xPars[1] },
          { "xParam2Final", xPars[2] },
          { "xParam3Final", xPars[3] },
          { "yParam0Init", ayPredicted },
          { "yParam1Init", byPredicted },
          { "xParam1Init", bxPredicted },
          { "nHits", static_cast<int>( track.size() ) },
      } };
      m_debugTool->storeData( vars_and_values, "MVAInput" );
    }
  }

  /**
   * @brief select and add stereo hits to make a full candidate out of found x candidates
   * @param candidates container for found x candidates
   * @param xPars x parameter bundle
   * @param cache cache of zone positions
   * @param veloSeed useful info from the input track
   * @param SciFiHits PrSciFiHits container
   * @return a pair of bools, indicating if there's a good and/or a ok-ish track
   * @details A loop over all previously found x candidates is performed. For each x candidate stereo hits are
   * collected by extrapolating the track to the stereo layers. From these hits stereo candidates are constructed
   * the best of which is selected and added to the x candidate to form a full candidate. The full candidate is fitted
   * again to determine the final track parameters, followed by a neural network assigning a ghost quality. The first
   * return value is true if there's a track with MORE than the minimum number of hits, the second is true if there's
   * any valid track.
   */
  template <typename TrackType>
  template <bool secondLoop, typename Buffer>
  auto PrForwardTracking<TrackType>::selectFullCandidates( LHCb::span<PrForwardTrack> candidates, PrParametersX xPars,
                                                           const ZoneCache& cache, const VeloSeedExtended& veloSeed,
                                                           const FT::Hits& SciFiHits, float magScaleFactor,
                                                           Buffer& counter ) const {

    if constexpr ( secondLoop )
      direct_debug( "------------selectFullCandidates------------secondLoop" );
    else
      direct_debug( "------------selectFullCandidates------------" );
    direct_debug( "" );
    PrParametersY yPars{ m_maxTolY,       m_uvSearchBinWidth,   m_tolY,          m_tolYSlope,
                         m_tolYUVSearch,  m_tolYTriangleSearch, m_minStereoHits, m_maxChi2StereoLinear,
                         m_maxChi2Stereo, m_maxChi2StereoAdd,   m_tolYMag,       m_tolYMagSlope };
    auto          anyGood{ false };
    auto          anyOK{ 0 };
    const auto    minQuality = std::isnan( veloSeed.seed.qOverP ) ? m_minQuality : m_minQualityMomentum;
    for ( auto& track : candidates ) {
      direct_debug( "++++ New XCandidate ++++" );
      xPars.minStereoHits = m_minStereoHits;
      xPars.maxChi2PerDoF = m_maxChi2PerDoFFinal;
      yPars.minStereoHits = m_minStereoHits;
      if ( track.size() + xPars.minStereoHits < m_minTotalHits ) {
        xPars.minStereoHits = m_minTotalHits - track.size();
        yPars.minStereoHits = m_minTotalHits - track.size();
      }

      if ( !selectStereoHits( track, SciFiHits, yPars, cache, veloSeed ) ) continue;

      // make a fit of all hits
      if ( !fitXProjection( track, xPars, SciFiHits ) ) continue;

      if ( fillEmptyXPlanes<secondLoop>( track, xPars, veloSeed, SciFiHits, m_maxChi2XAddFull ) ) {
        if ( !fitXProjection( track, xPars, SciFiHits ) ) continue;
      }
      direct_debug( "Track after final x fit of all hits:", track );
      assert( [&] {
        std::sort( track.getCoordsToFit().begin(), track.getCoordsToFit().end() );
        return std::adjacent_find( track.getCoordsToFit().begin(), track.getCoordsToFit().end() ) ==
               track.getCoordsToFit().end();
      }() );
      if ( m_debugTool.isEnabled() ) { calcMVAInput<true>( veloSeed, track, magScaleFactor ); }
      if ( track.size() >= m_minTotalHits ) {
        auto getQuality = [&] {
          const auto mvaInput = calcMVAInput<>( veloSeed, track, magScaleFactor );
          if constexpr ( std::is_same_v<TrackType, TracksUT> ) {
            const auto span = LHCb::span{ mvaInput }.subspan( 1 );
            const auto preselection =
                span[1] < m_maxDistX && span[2] < m_maxDistY && span[3] < m_maxY0Diff && span[4] < m_maxYSlopeDiff;
            if ( preselection ) {
              return std::isnan( veloSeed.seed.qOverP ) ? m_NN.GetMvaValue( LHCb::span<const float, 9>{ span } )
                                                        : m_NNVeloUT.GetMvaValue( mvaInput );
            } else {
              return 0.f;
            }

          } else {
            // check that certain inputs are within expected range
            const auto preselection = mvaInput[1] < m_maxDistX && mvaInput[2] < m_maxDistY &&
                                      mvaInput[3] < m_maxY0Diff && mvaInput[4] < m_maxYSlopeDiff;
            return preselection ? m_NN.GetMvaValue( mvaInput ) : 0.f;
          }
        };
        const auto quality = getQuality();
        direct_debug( "Ghost MVA response", quality, "> minQuality", minQuality );
        if ( quality > minQuality ) {
          track.setQuality( quality );
          track.setValid();
          ++anyOK;
          anyGood = track.size() > m_minTotalHits;
        }
      }
      direct_debug( "" );
    }
    counter += anyOK;
    return std::pair<bool, bool>{ anyGood, anyOK };
  }

  /**
   * @brief Uses a parameterisation to define the initial hit search window for an input track.
   * @param seed Information about direction of the input track.
   * @return A pair of x values corresponding to the search window borders at the reference plane.
   * @details The basic search window defines borders at the reference plane based on the input track's
   * direction in x and y and a minimum momentum requirement (pt and p). The basic search window can be larger than the
   * dimensions of the detector if the momentum requirements are low. If a momentum estimate of the input track is
   * available (VeloUT), the search window can be narrowed by taking into account sign and magnitude of the momentum.
   */
  template <typename TrackType>
  auto PrForwardTracking<TrackType>::calculateMomentumBorders( const VeloSeed& seed ) const {

    const auto minInvPGeVfromPt  = float{ Gaudi::Units::GeV } / m_minPT * seed.momProj;
    const auto minInvPGeV        = float{ Gaudi::Units::GeV } / m_minP;
    const auto minPBorder        = seed.calcMomentumBorder( minInvPGeV );
    const auto minPTBorder       = seed.calcMomentumBorder( minInvPGeVfromPt );
    const auto minMomentumBorder = std::min( minPBorder, minPTBorder );

    auto dxMin = -minMomentumBorder;
    auto dxMax = minMomentumBorder;

    if constexpr ( std::is_same_v<TrackType, TracksUT> ) {
      if ( m_useMomentumSearchWindow && !std::isnan( seed.qOverP ) ) {

        const auto invPGeV = std::abs( seed.qOverP ) * float{ Gaudi::Units::GeV };
        const auto pBorder = seed.calcMomentumBorder( invPGeV );

        const auto lowerMomentumError = std::clamp( m_upperLimitOffset + m_upperLimitSlope * invPGeV,
                                                    m_upperLimitMin.value(), m_upperLimitMax.value() );
        const auto higherMomentumError =
            std::min( m_lowerLimitOffset + m_lowerLimitSlope * invPGeV, m_lowerLimitMax.value() );

        if ( const auto deflectionToPositiveX = seed.qMag < 0.f; deflectionToPositiveX ) {
          dxMin = std::max( 0.f, pBorder - higherMomentumError );
          dxMax = pBorder + lowerMomentumError;
        } else {
          dxMin = -pBorder - lowerMomentumError;
          dxMax = std::min( 0.f, -pBorder + higherMomentumError );
        }

        if ( const auto pt = std::abs( 1.f / seed.qOverP ) * seed.momProj;
             pt > m_wrongSignPT && m_useWrongSignWindow ) {
          dxMin = -pBorder - lowerMomentumError;
          dxMax = pBorder + lowerMomentumError;
        }
      }
    }
    return std::pair{ dxMin, dxMax };
  }

  /**
   * @brief remove duplicates from the final tracks
   * @param tracks all forwarded tracks
   * @param SciFiHits PrScFiHits container
   * @return vector of sorted LHCbIDS vector per track
   * @details This function is an important part of ghost and clone reduction. A duplicate is found if two tracks
   * share a certain fraction of hits. If one of the two tracks has a significantly lower quality (from ghost NN),
   * it is removed.
   */
  template <typename TrackType>
  auto PrForwardTracking<TrackType>::removeDuplicates( PrForwardTracks& tracks, const FT::Hits& SciFiHits ) const {

    const auto zEndT = StateParameters::ZEndT - zReference;
    std::sort( tracks.begin(), tracks.end(),
               [&]( auto const& t1, auto const& t2 ) { return t1.x( zEndT ) < t2.x( zEndT ); } );

    const auto                                                           nTotal = tracks.size();
    auto                                                                 nDuplicates{ 0 };
    std::vector<static_vector<LHCb::LHCbID, Detector::FT::nLayersTotal>> idSets( nTotal );
    for ( const auto& [idx, track] : LHCb::range::enumerate( tracks ) ) {
      const auto& coordsToFit = track.getCoordsToFit();
      auto&       idVec       = idSets[idx];
      std::transform( coordsToFit.begin(), coordsToFit.end(), std::back_inserter( idVec ),
                      [&SciFiHits]( auto iHit ) { return SciFiHits.lhcbid( iHit ); } );
      std::sort( idVec.begin(), idVec.end() );
    }
    // unsigned to fix warning
    for ( unsigned i1{ 0 }; i1 < nTotal; ++i1 ) {
      auto& t1 = tracks[i1];
      if ( !t1.valid() ) continue;
      for ( unsigned i2 = i1 + 1; i2 < nTotal; ++i2 ) {
        auto& t2 = tracks[i2];
        if ( !t2.valid() ) continue;
        // The distance only gets larger because we sorted above!
        if ( std::abs( t1.x( zEndT ) - t2.x( zEndT ) ) > m_minXInterspace ) break;
        if ( std::abs( t1.y( zEndT ) - t2.y( zEndT ) ) > m_minYInterspace ) continue;

        const auto nCommon = std::set_intersection( idSets[i1].begin(), idSets[i1].end(), idSets[i2].begin(),
                                                    idSets[i2].end(), CountIterator{} )
                                 .count();

        if ( 2.f * nCommon > m_maxCommonFrac * ( idSets[i1].size() + idSets[i2].size() ) ) {
          if ( const auto deltaQ = t2.quality() - t1.quality(); deltaQ < -m_deltaQuality ) {
            ++nDuplicates;
            t2.setInvalid();
          } else if ( deltaQ > m_deltaQuality ) {
            ++nDuplicates;
            t1.setInvalid();
          }
        }
      }
    }
    m_duplicateCnt += nDuplicates;
    return idSets;
  }

  /**
   * @brief converts internal track representation to general long tracks and adds UT hits
   * @param tracks contains found internal tracks
   * @param idSets contains the LHCbIDs for each track
   * @param input_tracks the input track that was forwarded
   * @details It's the only place where LHCbIds come into play because they are needed elsewhere.
   */
  template <typename TrackType>
  template <typename Container>
  auto PrForwardTracking<TrackType>::makeLHCbLongTracks( PrForwardTracks&& tracks, Container&& idSets,
                                                         const TrackType&        input_tracks,
                                                         const IPrAddUTHitsTool& addUTHitsTool ) const {

    TracksFT result = make_TracksFT_from_ancestors( input_tracks );
    result.reserve( tracks.size() );

    auto const inputtracks = input_tracks.scalar();
    for ( auto&& [cand, ids] : Gaudi::Functional::details::zip::range( tracks, idSets ) ) {
      if ( !cand.valid() ) continue;
      const auto iTrack      = cand.track();
      const auto ancestTrack = inputtracks[iTrack];

      auto n_vphits = scalar::int_v{ 0 };
      auto n_uthits = scalar::int_v{ 0 };

      auto out = result.emplace_back<SIMDWrapper::InstructionSet::Scalar>();
      out.field<TracksTag::trackSeed>().set( -1 );

      if constexpr ( std::is_same_v<TrackType, TracksUT> ) {
        n_vphits = ancestTrack.nVPHits();
        n_uthits = ancestTrack.nUTHits();
        out.field<TracksTag::trackVP>().set( ancestTrack.trackVP() );
        out.field<TracksTag::trackUT>().set( iTrack );
        out.field<TracksTag::VPHits>().resize( n_vphits );
        out.field<TracksTag::UTHits>().resize( n_uthits );

        for ( auto idx{ 0 }; idx < n_vphits.cast(); ++idx ) {
          out.field<TracksTag::VPHits>()[idx].template field<TracksTag::Index>().set( ancestTrack.vp_index( idx ) );
          out.field<TracksTag::VPHits>()[idx].template field<TracksTag::LHCbID>().set( ancestTrack.vp_lhcbID( idx ) );
        }
        for ( auto idx{ 0 }; idx < n_uthits.cast(); ++idx ) {
          out.field<TracksTag::UTHits>()[idx].template field<TracksTag::Index>().set( ancestTrack.ut_index( idx ) );
          out.field<TracksTag::UTHits>()[idx].template field<TracksTag::LHCbID>().set( ancestTrack.ut_lhcbID( idx ) );
        }
      } else {
        n_vphits = ancestTrack.nHits();
        out.field<TracksTag::trackVP>().set( iTrack );
        out.field<TracksTag::trackUT>().set( -1 );
        out.field<TracksTag::VPHits>().resize( n_vphits );
        out.field<TracksTag::UTHits>().resize( 0 );

        for ( auto idx{ 0 }; idx < n_vphits.cast(); ++idx ) {
          out.field<TracksTag::VPHits>()[idx].template field<TracksTag::Index>().set( ancestTrack.vp_index( idx ) );
          out.field<TracksTag::VPHits>()[idx].template field<TracksTag::LHCbID>().set( ancestTrack.vp_lhcbID( idx ) );
        }
      }

      static_assert( Event::v3::num_states<Event::Enum::Track::Type::Long>() == 2 );
      constexpr auto EndVelo = Event::Enum::State::Location::EndVelo;
      constexpr auto endT    = stateIndex<Event::Enum::Track::Type::Long>( Event::Enum::State::Location::EndT );
      constexpr auto zEndT   = static_cast<float>( Z( Event::Enum::State::Location::EndT ).value() );
      constexpr auto dz      = zEndT - zReference;

      const auto qOverP = cand.getQoP();
      const auto pos    = LHCb::LinAlg::Vec<scalar::float_v, 3>{ cand.x( dz ), cand.y( dz ), zEndT };
      const auto dir    = LHCb::LinAlg::Vec<scalar::float_v, 3>{ cand.xSlope( dz ), cand.ySlope( dz ), 1.f };
      out.field<TracksTag::States>( endT ).setPosition( pos );
      out.field<TracksTag::States>( endT ).setDirection( dir );
      out.field<TracksTag::States>( endT ).setQOverP( qOverP );

      const auto [ancestpos, ancestdir] = [&] {
        if constexpr ( std::is_same_v<TrackType, TracksUT> ) {
          const auto velozipped  = input_tracks.getVeloAncestors()->scalar();
          const auto trackVP     = ancestTrack.trackVP();
          const auto velo_scalar = velozipped[trackVP.cast()];
          return std::pair{ velo_scalar.StatePos( EndVelo ), velo_scalar.StateDir( EndVelo ) };
        } else {
          return std::pair{ ancestTrack.StatePos( EndVelo ), ancestTrack.StateDir( EndVelo ) };
        }
      }();
      constexpr auto endVeloIndex = stateIndex<Event::Enum::Track::Type::Long>( EndVelo );
      out.field<TracksTag::States>( endVeloIndex ).setPosition( ancestpos );
      out.field<TracksTag::States>( endVeloIndex ).setDirection( ancestdir );
      out.field<TracksTag::States>( endVeloIndex ).setQOverP( qOverP );

      assert( ids.size() <= LHCb::Pr::Long::Tracks::MaxFTHits );

      const auto n_fthits = ids.size();
      out.field<TracksTag::FTHits>().resize( n_fthits );
      const auto& hits = cand.getCoordsToFit();
      // unsigned to avoid warning
      for ( unsigned idx{ 0 }; idx < n_fthits; ++idx ) {
        out.field<TracksTag::FTHits>()[idx].template field<TracksTag::Index>().set( hits[idx] );
        out.field<TracksTag::FTHits>()[idx].template field<TracksTag::LHCbID>().set( ids[idx] );
      }
    } // next candidate

    //=== add UT hits
    addUTHitsTool.addUTHits( result );

    m_outputTracksCnt += result.size();
    direct_debug( "" );
    direct_debug( "Bye Bye, track you soon! <3" );
    direct_debug( "" );
    return result;
  }

} // namespace LHCb::Pr::Forward
