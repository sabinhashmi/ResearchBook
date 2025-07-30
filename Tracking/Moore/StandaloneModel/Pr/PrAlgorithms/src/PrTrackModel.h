/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#pragma once

#include "Core/FloatComparison.h"
#include "Detector/FT/FTConstants.h"
#include "Event/PrHits.h"
#include "Event/SOACollection.h"
#include "LHCbMath/SIMDWrapper.h"
#include "PrKernel/FTGeometryCache.h"
#include "boost/container/static_vector.hpp"
#include <array>
#include <bitset>
#include <iostream>
#include <limits>

namespace LHCb::Pr {

  /**
   * @brief Class to calculate y correction for velo track and magnet kink position seen from EndT.
   *
   * @tparam F Underlying floating point implementation, usually float or simd::float_v.
   *
   * @details This class is introduced to easily share the parameterisiations related to the matching
   * of a Velo track to a SciFi extensions in PrForwardTracking and PrMatchNN. The parameterisations
   * can be obtained using the Reco-Parameterisiation-Tuner.
   * https://gitlab.cern.ch/gunther/prforwardtracking-parametrisation-tuner/-/tree/master
   */
  template <typename F>
  struct VeloSciFiMatch {

    VeloSciFiMatch( F tx, F ty, F tx2, F ty2 ) {
      xTerm1   = bendYParams[3] * ty * tx + bendYParams[8] * ty * ty2 * tx;
      xTerm2   = bendYParams[4] * ty;
      xTerm3   = bendYParams[1] * ty + bendYParams[7] * ty * tx2;
      yTerm1   = bendYParams[0] + bendYParams[6] * tx2;
      yTerm2   = bendYParams[5] * ty;
      yTerm3   = bendYParams[2] * ty;
      yTerm1M  = bendYParamsMatch[0] * ty;
      yTerm2M  = bendYParamsMatch[1] * ty;
      zMagTerm = zMagnetParamsEndT[0] + zMagnetParamsEndT[3] * tx * tx;
    }

    /**
     * @brief Calculate (simple) estimate of the magnet kink z position for PrMatchNN.
     *
     * @param dSlopeAbs Absolute difference between x slope at ZEndT and Velo track x slope (tx_endT - tx).
     * @return Estimate of the magnet kink z position.
     *
     * @details It is not unimportant to use the dSlope from ZEndT here because this is (roughly, actually z position
     * of the last SciFi layer) where the parameterisiation was derived, and zMag shifts depending on where
     * dSlope is deterimined because of the fringe magnetic fields.
     */
    auto calcZMagEndT( F dSlopeAbs, F xEndT ) const {
      return zMagTerm + dSlopeAbs * ( zMagnetParamsEndT[1] + zMagnetParamsEndT[4] * dSlopeAbs ) +
             zMagnetParamsEndT[2] * abs( xEndT );
    }

    /**
     * @brief Calculate estimate of difference between Velo track y straight line extrapolation to ZEndT and true y
     * coord.
     *
     * @param dSlope Difference between x slope at ZEndT and Velo track x slope (tx_endT - tx).
     * @param dSlopeY Difference between y slope at ZEndT and Velo track y slope (ty_endT - ty).
     * @return Correction to apply to y straight line extrapolation of Velo track (i.e yStraightAtEndT + yCorr).
     */
    auto calcYCorr( F dSlope, F dSlopeY ) const {
      return dSlopeY * ( yTerm1 + yTerm2 * dSlopeY ) + dSlope * ( xTerm2 * dSlope + xTerm1 ) + abs( dSlope ) * xTerm3 +
             abs( dSlopeY ) * yTerm3;
    }
    /**
     * @brief Simplified estimate of difference between velo track and y straight line extrapolation for PrMatchNN.
     *
     * @param dSlope2 Squared difference between x slope at ZEndT and Velo track x slope (tx_endT - tx).
     * @param dSlopeY2 Squared difference between y slope at ZEndT and Velo track y slope (ty_endT - ty).
     * @return Correction to apply to y straight line extrapolation of Velo track (i.e yStraightAtEndT + yCorr).
     */
    auto calcYCorrMatch( F dSlope2, F dSlopeY2 ) const { return yTerm1M * dSlope2 + yTerm2M * dSlopeY2; }

  private:
    F xTerm1;
    F xTerm2;
    F xTerm3;
    F yTerm1;
    F yTerm2;
    F yTerm3;
    F yTerm1M;
    F yTerm2M;
    F zMagTerm;

    // param[0] + param[1]*dSlope_xEndT_abs + param[2]*x_EndT_abs + param[3]*tx^2 +
    // param[4]*dSlope_xEndT^2
    static constexpr std::array<float, 5> zMagnetParamsEndT{
        5286.687877988849f, -3.259689996453795f, 0.015615778872337033f, -1377.3175211789967f, 282.9821232487341f };

    /*  param[0]*dSlope_yEndT + param[1]*ty dSlope_xEndT_abs + param[2]*ty
     *  dSlope_yEndT_abs + param[3]*ty tx dSlope_xEndT + param[4]*ty dSlope_xEndT^2 +
     *  param[5]*ty dSlope_yEndT^2 + param[6]*tx^2 dSlope_yEndT + param[7]*ty tx^2
     *  dSlope_xEndT_abs + param[8]*ty^3 tx dSlope_xEndT
     */
    static constexpr std::array<float, 9> bendYParams{ 4227.700246518678f,  49.57143777936437f,   1771.7315744432228f,
                                                       1939.3072869085627f, 1536.7479670346445f,  287.1741842353233f,
                                                       1901.2686457070686f, -1757.6881109430192f, 17703.99684842123f };
    // param[0]*ty dSlope_xEndT^2 + param[1]*ty dSlope_yEndT^2
    static constexpr std::array<float, 2> bendYParamsMatch{ -1974.6355416889746f, -35933.837494833504f };
  };

  namespace Forward {

    using boost::container::static_vector;
    using ZoneCache                   = Detector::FT::Cache::GeometryCache;
    inline constexpr auto nan         = std::numeric_limits<float>::signaling_NaN();
    inline constexpr auto nanMomentum = std::numeric_limits<float>::quiet_NaN();

    inline constexpr float zReference{ 8520.f };
    // this must be false in production!
    inline constexpr auto DEBUG = false;

    template <bool enable = DEBUG, typename... Args>
    constexpr void direct_debug( Args&&... args ) {
      if constexpr ( enable ) { ( ( std::cout << std::forward<Args>( args ) << " " ), ... ) << std::endl; }
    }

    template <typename Tuple, size_t... I>
    constexpr void tuple_helper( Tuple&& tuple, std::index_sequence<I...> ) {
      ( ( std::cout << std::get<I>( tuple ) << " " ), ... ) << std::endl;
    }

    template <bool enable = DEBUG, typename... T>
    constexpr void debug_tuple( std::tuple<T...>&& tuple ) {
      if constexpr ( enable ) { tuple_helper( tuple, std::make_index_sequence<sizeof...( T )>() ); }
    }

    /**
     * @struct TrackPars PrForwardTrack.h
     * @brief aggregate for storage and handling of track parameters
     */
    template <auto P>
    struct TrackPars : std::array<float, P> {
      template <auto N>
      void add( LHCb::span<const float, N> v ) {
        static_assert( N <= P );
        for ( size_t i{ 0 }; i < N; ++i ) { this->operator[]( i ) += v[i]; }
      }

      // for debugging
      friend std::ostream& operator<<( std::ostream& os, const TrackPars& p ) {
        os << "[ ";
        for ( auto par : p ) { os << par << " "; }
        os << "]";
        return os;
      }
    };

    /**
     * @brief Wrapper around static_vector holding fit coordinates.
     *
     * @tparam T type of coordinate e.g. int
     * @tparam N maximum number of coordinates e.g. number of FT layers.
     */
    template <typename T, auto N>
    struct FitCoords : static_vector<T, N> {
      using static_vector<T, N>::static_vector;
      // for debugging
      friend std::ostream& operator<<( std::ostream& os, const FitCoords& p ) {
        os << "[ ";
        for ( auto coord : p ) { os << coord << " "; }
        os << "]";
        return os;
      }
    };

    /**
     * @brief Class to add and bundle useful parameters of the Velo track.
     *
     * @details Calculates terms needed to estimate the momentum borders at construction.
     * The parameterisations used for this can be obtained using the Reco-Parameterisiation-Tuner.
     * https://gitlab.cern.ch/gunther/prforwardtracking-parametrisation-tuner/-/tree/master
     */
    struct VeloSeed {

      VeloSeed() = default;
      VeloSeed( float x0, float y0, float z0, float tx, float ty, float qOverP, float magScaleFactor )
          : x0{ x0 }
          , y0{ y0 }
          , z0{ z0 }
          , tx{ tx }
          , ty{ ty }
          , qOverP{ qOverP }
          , tx2{ tx * tx }
          , ty2{ ty * ty }
          , slope2{ tx * tx + ty * ty }
          , momProj{ std::sqrt( ( tx * tx + ty * ty ) / ( 1.f + tx * tx + ty * ty ) ) } {

        qMag       = !std::isnan( qOverP ) ? magScaleFactor * std::copysign( 1.f, qOverP ) : std::abs( magScaleFactor );
        linearTerm = momentumWindowParamsRef[0] + momentumWindowParamsRef[1] * ty2 + momentumWindowParamsRef[2] * tx2 +
                     momentumWindowParamsRef[5] * tx2 * tx * qMag + momentumWindowParamsRef[9] * ty2 * tx2 +
                     momentumWindowParamsRef[12] * tx2 * tx2;
        quadraticTerm = momentumWindowParamsRef[3] * tx * qMag + momentumWindowParamsRef[6] * tx2 +
                        momentumWindowParamsRef[10] * ty2 * tx * qMag;
        cubicTerm =
            momentumWindowParamsRef[4] + momentumWindowParamsRef[7] * tx * qMag + momentumWindowParamsRef[11] * ty2;
      }

      float x0{ nan };
      float y0{ nan };
      float z0{ nan };
      float tx{ nan };
      float ty{ nan };
      float qOverP{ nan };
      float tx2{ nan };
      float ty2{ nan };
      float slope2{ nan };
      float momProj{ nan };
      float qMag{ nan };
      float linearTerm{ nan };
      float quadraticTerm{ nan };
      float cubicTerm{ nan };

      /**
       * @brief Linear extrapolation of the Velo track in x.
       *
       * @param z Position to extrapolate to.
       * @return x coordinate at z.
       */
      auto x( float z ) const { return x0 + ( z - z0 ) * tx; }

      /**
       * @brief Linear extrapolation of the Velo track in y.
       *
       * @param z Position to extrapolate to.
       * @return y coordinate at z.
       */
      auto y( float z ) const { return y0 + ( z - z0 ) * ty; }

      /**
       * @brief Estimate the x difference to straight line extrapolation due to momentum.
       *
       * @param invPGeV One over momentum in GeV.
       * @return x difference to straight line extrapolation.
       */
      auto calcMomentumBorder( float invPGeV ) const {
        return invPGeV * ( linearTerm + invPGeV * ( quadraticTerm +
                                                    invPGeV * ( cubicTerm + momentumWindowParamsRef[8] * invPGeV ) ) );
      }

      // consider VeloSeeds equal when state at EndVelo is equal
      bool essentiallyEqual( const VeloSeed& other ) const {
        return LHCb::essentiallyEqual( x0, other.x0 ) && LHCb::essentiallyEqual( y0, other.y0 ) &&
               LHCb::essentiallyEqual( tx, other.tx ) && LHCb::essentiallyEqual( ty, other.ty );
      }

      // for debugging
      friend std::ostream& operator<<( std::ostream& os, const VeloSeed& v ) {
        return os << "VeloSeed: "
                  << "[x0,y0,z0,tx,ty,qOverP] = "
                  << "[ " << v.x0 << ", " << v.y0 << ", " << v.z0 << ", " << v.tx << ", " << v.ty << ", " << v.qOverP
                  << " ]";
      }

      // param[0]*inv_p_gev + param[1]*ty^2 inv_p_gev + param[2]*tx^2 inv_p_gev +
      // param[3]*tx inv_p_gev pol_qop_gev + param[4]*inv_p_gev^3 + param[5]*tx^3
      // pol_qop_gev + param[6]*tx^2 inv_p_gev^2 + param[7]*tx inv_p_gev^2 pol_qop_gev
      // + param[8]*inv_p_gev^4 + param[9]*ty^2 tx^2 inv_p_gev + param[10]*ty^2 tx
      // inv_p_gev pol_qop_gev + param[11]*ty^2 inv_p_gev^3 + param[12]*tx^4 inv_p_gev
      static constexpr std::array<float, 13> momentumWindowParamsRef{
          4018.896625676043f,  6724.789549369031f,  3970.9093976497766f, -4363.5807241252905f, 1421.1056758688073f,
          4934.07761471779f,   6985.252911263751f,  -5538.28013195104f,  1642.8616070452542f,  106068.96918885755f,
          -94446.81037767915f, 26489.793756692892f, -23936.54391006025f };
    };

    /**
     * @brief Class that bundles parameters and values used to extend a Velo Track into SciFi.
     *
     * @details This class is often passed to functions because it contains many useful quantities
     * that would have been passed separately otherwise, pollution the function signatures. On
     * construction, terms used to estimate expected values of the SciFi extension are calculated
     * from Velo track quantities. The parameterisations used for this expectation can be obtained
     * using the Reco-Parameterisiation-Tuner.
     * https://gitlab.cern.ch/gunther/prforwardtracking-parametrisation-tuner/-/tree/master
     *
     */
    struct VeloSeedExtended {
      VeloSeedExtended( int iTrack, const VeloSeed& veloseed, const ZoneCache& cache )
          : veloSciFiMatch{ veloseed.tx, veloseed.ty, veloseed.tx2, veloseed.ty2 }
          , seed{ veloseed }
          , xStraightAtRef{ veloseed.x( zReference ) }
          , zMag{ zMagnetParamsRef[0] + zMagnetParamsRef[1] * veloseed.tx2 + zMagnetParamsRef[3] * veloseed.ty2 }
          , yStraightAtRef{ veloseed.y( zReference ) }
          , iTrack{ iTrack } {

        const auto tx     = seed.tx;
        const auto ty     = seed.ty;
        const auto tx2    = seed.tx2;
        const auto ty2    = seed.ty2;
        const auto tx3    = tx2 * tx;
        const auto ty3    = ty2 * ty;
        const auto txty   = tx * ty;
        const auto tytx2  = ty * tx2;
        const auto tytx3  = ty * tx3;
        const auto ty3tx  = ty3 * tx;
        const auto ty3tx2 = ty3 * tx2;

        upperHalfTrack = yStraightAtRef > 0.f;
        pointingSide =
            xStraightAtRef > 0.f ? LHCb::Detector::FTChannelID::Side::A : LHCb::Detector::FTChannelID::Side::C;
        zMagTerm = zMagnetParamsRef[2] * tx;
        cxTerm   = cxParams[0] + cxParams[1] * tx + cxParams[2] * ty + cxParams[3] * tx2 + cxParams[4] * txty +
                 cxParams[5] * ty2;
        dxTerm = dxParams[0] + dxParams[1] * tx + dxParams[2] * ty + dxParams[3] * tx2 + dxParams[4] * txty +
                 dxParams[5] * ty2;
        yCorrTerm =
            yCorrParamsRef[0] + yCorrParamsRef[2] * txty + yCorrParamsRef[5] * ty3tx + yCorrParamsRef[6] * tytx3;
        yCorrTermAbs =
            yCorrParamsRef[1] * ty + yCorrParamsRef[3] * ty3 + yCorrParamsRef[4] * tytx2 + yCorrParamsRef[7] * ty3tx2;
        tyCorrTerm    = tyCorrParamsRef[0] * txty + tyCorrParamsRef[4] * tytx3;
        tyCorrTermAbs = tyCorrParamsRef[2] * ty3 + tyCorrParamsRef[3] * tytx2 + tyCorrParamsRef[5] * ty3tx2;
        cyTerm        = cyParams[1] * txty;
        cyTermAbs     = cyParams[0] * ty + cyParams[3] * ty3 + cyParams[4] * tytx2;

        for ( unsigned int iLayer{ 0 }; iLayer < Detector::FT::nLayersTotal; ++iLayer ) {
          if ( iLayer % 2 ) {
            const auto l      = iLayer / 2;
            yCorrTermLayer[l] = yCorrParamsLayers[l][0] + yCorrParamsLayers[l][2] * txty +
                                yCorrParamsLayers[l][5] * ty3tx + yCorrParamsLayers[l][6] * tytx3;
            yCorrTermAbsLayer[l] = yCorrParamsLayers[l][1] * ty + yCorrParamsLayers[l][3] * ty3 +
                                   yCorrParamsLayers[l][4] * tytx2 + yCorrParamsLayers[l][7] * ty3tx2;
          }
          const auto zLayer           = cache.z( iLayer );
          const auto yStraightInLayer = seed.y( zLayer );
          betterZ[iLayer]             = zLayer + yStraightInLayer * cache.dzdy( iLayer );
          xStraightInZone[iLayer]     = seed.x( betterZ[iLayer] );
          yStraightInZone[iLayer]     = seed.y( betterZ[iLayer] );
        }
      }

      VeloSeedExtended() = delete;

      /**
       * @brief Estimate magnet kink z position.
       *
       * @tparam F Floating point type, usually float or simd::float_v.
       * @param dSlope Difference between x slope in SciFi and Velo track (tx_ref - tx).
       * @return Estimated z position of the magnet kink.
       *
       * @details The parameterisation was found using dSlope calculated at the reference plane.
       * Because of fringe magnetic fields, zMag can be different depending on which x slope in the SciFi
       * is used.
       */
      template <typename F>
      auto calcZMag( F dSlope ) const {
        return zMag + ( zMagTerm + zMagnetParamsRef[4] * dSlope ) * dSlope;
      }

      /**
       * @brief Estimate quadratic coefficient of x track model.
       *
       * @tparam F Floating point type, usually float or simd::float_v.
       * @param dSlope Difference between x slope in SciFi and Velo track (tx_ref - tx).
       * @return Quadratic coefficient of x track model.
       */
      template <typename F>
      auto calcCX( F dSlope ) const {
        return cxTerm * dSlope;
      }

      /**
       * @brief Estimate cubic coefficient of x track model.
       *
       * @tparam F Floating point type, usually float or simd::float_v.
       * @param dSlope Difference between x slope in SciFi and Velo track (tx_ref - tx).
       * @return Cubic coefficient of x track model.
       */
      template <typename F>
      auto calcDX( F dSlope ) const {
        return dxTerm * dSlope;
      }

      /**
       * @brief Estimate x position correction to straight line.
       *
       * @tparam F Floating point type, usually float or simd::float_v.
       * @param dSlope (Estimated) Difference between x slope in SciFi and Velo track.
       * @param dz z distance between layer/hit and reference plane.
       * @param dz2 dz squared.
       * @return Correction to x position of hit, such that hit falls on straight line.
       *
       * @details This must be understood within the optical track model of the
       * Forward tracking, where the trajectories are treated the same as light
       * refraction in a thin lense and thus as straight lines.
       */
      template <typename F>
      auto calcXCorr( F dSlope, float dz, float dz2 ) const {
        return dSlope * dz2 * ( cxTerm + dz * dxTerm );
      }

      /**
       * @brief Estimate difference between linear extrapolation in y and true position.
       *
       * @tparam F Floating point type, usually float or simd::float_v.
       * @param dSlope (Estimated) Difference between x slope in SciFi and Velo track.
       * @param l Index of SciFi layer.
       * @return Difference between linear extrapolation and true y position for given layer.
       */
      template <typename F>
      auto calcYCorr( F dSlope, int l ) const {
        l /= 2u;
        return yCorrTermLayer[l] * dSlope + yCorrTermAbsLayer[l] * abs( dSlope );
      }

      /**
       * @brief Estimate difference between linear extrapolation in y and true position.
       *
       * @tparam F Floating point type, usually float or simd::float_v.
       * @param dSlope Difference between x slope in SciFi and Velo track (tx_ref - tx).
       * @return Difference between linear extrapolation and true y position at reference plane.
       */
      template <typename F>
      auto calcYCorr( F dSlope ) const {
        return yCorrTerm * dSlope + yCorrTermAbs * abs( dSlope );
      }

      /**
       * @brief Estimate correction to y slope of velo track slope after magnet.
       *
       * @tparam F Floating point type, usually float or simd::float_v.
       * @param dSlope Difference between x slope in SciFi and Velo track (tx_ref - tx).
       * @return Correction to velo track's y slope at reference plane.
       */
      template <typename F>
      auto calcTyCorr( F dSlope ) const {
        return tyCorrTermAbs * abs( dSlope ) + ( tyCorrTerm + tyCorrParamsRef[1] * seed.ty * dSlope ) * dSlope;
      }

      /**
       * @brief Estimate quadratic coefficient of y track model.
       *
       * @tparam F Floating point type, usually float or simd::float_v.
       * @param dSlope Difference between x slope in SciFi and Velo track (tx_ref - tx).
       * @return Quadratic coefficient of y track model.
       */
      template <typename F>
      auto calcCY( F dSlope ) const {
        return cyTermAbs * abs( dSlope ) + ( cyTerm + cyParams[2] * seed.ty * dSlope ) * dSlope;
      }

      // for debugging
      friend std::ostream& operator<<( std::ostream& os, const VeloSeedExtended& v ) {
        os << "VeloSeedExtended: "
           << "goes through upper half = " << v.upperHalfTrack << std::endl;
        os << "xStraightatRef = " << v.xStraightAtRef << ", "
           << "zMag = " << v.zMag << ", "
           << "yStraightAtRef = " << v.yStraightAtRef;
        return os;
      }

      std::array<float, Detector::FT::nLayersTotal>   betterZ;
      std::array<float, Detector::FT::nLayersTotal>   xStraightInZone;
      std::array<float, Detector::FT::nLayersTotal>   yStraightInZone;
      std::array<float, Detector::FT::nUVLayersTotal> yCorrTermLayer;
      std::array<float, Detector::FT::nUVLayersTotal> yCorrTermAbsLayer;
      VeloSciFiMatch<float>                           veloSciFiMatch;
      const VeloSeed&                                 seed;
      float                                           zMagTerm;
      float                                           cxTerm;
      float                                           dxTerm;
      float                                           yCorrTerm;
      float                                           yCorrTermAbs;
      float                                           tyCorrTerm;
      float                                           tyCorrTermAbs;
      float                                           cyTerm;
      float                                           cyTermAbs;
      float                                           xStraightAtRef;
      float                                           zMag;
      float                                           yStraightAtRef;
      int                                             iTrack;
      LHCb::Detector::FTChannelID::Side               pointingSide;
      bool                                            upperHalfTrack;

      // param[0] + param[1]*tx^2 + param[2]*tx dSlope_fringe + param[3]*ty^2 +
      // param[4]*dSlope_fringe^2
      static constexpr std::array<float, 5> zMagnetParamsRef{
          5205.144186525624f, -320.7206595710594f, 702.1384894815535f, -316.36350963107543f, 441.59909857558097f };

      // param[0]*dSlope_fringe + param[1]*tx dSlope_fringe + param[2]*ty
      // dSlope_fringe + param[3]*tx^2 dSlope_fringe + param[4]*tx ty dSlope_fringe +
      // param[5]*ty^2 dSlope_fringe
      static constexpr std::array<float, 6> cxParams{ 2.335283084724005e-05f,   -5.394341220986507e-08f,
                                                      -1.1353152524130453e-06f, 9.213281616649267e-06f,
                                                      -6.76457896718169e-07f,   -0.0003740758569392804f };
      // param[0]*dSlope_fringe + param[1]*tx dSlope_fringe + param[2]*ty
      // dSlope_fringe + param[3]*tx^2 dSlope_fringe + param[4]*tx ty dSlope_fringe +
      // param[5]*ty^2 dSlope_fringe
      static constexpr std::array<float, 6> dxParams{ -7.057523874477465e-09f, 1.0524178059699073e-11f,
                                                      6.46124765440666e-10f,   2.595690034874298e-09f,
                                                      8.044356540608104e-11f,  9.933758467661586e-08f };
      // param[0]*dSlope_fringe + param[1]*ty dSlope_fringe_abs + param[2]*ty tx
      // dSlope_fringe + param[3]*ty^3 dSlope_fringe_abs + param[4]*ty tx^2
      // dSlope_fringe_abs + param[5]*ty^3 tx dSlope_fringe + param[6]*ty tx^3
      // dSlope_fringe + param[7]*ty^3 tx^2 dSlope_fringe_abs
      static constexpr std::array<std::array<float, 8>, 6> yCorrParamsLayers{
          { { 1.9141402652138315f, 154.61935746400832f, 3719.298754021463f, -6981.575944838166f, -67.7612042340458f,
              41484.88865215446f, 30544.717526101966f, 211219.00520598015f },
            { 1.9802106454737567f, 146.34197177414035f, 3766.9995843145575f, -7381.001822418669f, 18.407833054380728f,
              42635.398541425144f, 31434.95400997568f, 218404.36150766257f },
            { 2.6036680178541256f, 53.231282135657125f, 4236.335446831202f, -10844.798302911375f, 986.1498917330866f,
              52670.269097485856f, 39380.4857744525f, 281250.90766092145f },
            { 2.6802443731107797f, 40.75834605688442f, 4296.645356936966f, -11234.776424245354f, 1115.363228090216f,
              53813.817216417505f, 40299.07624778942f, 288431.507847565f },
            { 3.3827128857688793f, -76.61325300322648f, 4875.424130053332f, -14585.199358667853f, 2322.162251501158f,
              63618.048819648175f, 48278.83901554796f, 350657.56046107266f },
            { 3.4657288815375846f, -90.58976402034898f, 4946.538479838353f, -14962.319670402725f, 2464.758450826609f,
              64707.51942328425f, 49179.43246319585f, 357681.17176708044f } } };
      // param[0]*dSlope_fringe + param[1]*ty dSlope_fringe_abs + param[2]*ty tx
      // dSlope_fringe + param[3]*ty^3 dSlope_fringe_abs + param[4]*ty tx^2
      // dSlope_fringe_abs + param[5]*ty^3 tx dSlope_fringe + param[6]*ty tx^3
      // dSlope_fringe + param[7]*ty^3 tx^2 dSlope_fringe_abs
      static constexpr std::array<float, 8> yCorrParamsRef{ 2.5415524238347658f, 63.25841388467006f, 4187.534822693825f,
                                                            -10520.25391602297f, 881.6859925052617f, 51730.04107647908f,
                                                            38622.50428524951f,  275325.5721020971f };
      // param[0]*ty tx dSlope_fringe + param[1]*ty dSlope_fringe^2 + param[2]*ty^3
      // dSlope_fringe_abs + param[3]*ty tx^2 dSlope_fringe_abs + param[4]*ty tx^3
      // dSlope_fringe + param[5]*ty^3 tx^2 dSlope_fringe_abs
      static constexpr std::array<float, 6> tyCorrParamsRef{ 0.9346197967408639f, -0.4658007458482092f,
                                                             -4.119808929050499f, 2.9514781492224613f,
                                                             12.5961355543964f,   39.98472114588754f };
      // param[0]*ty dSlope_fringe_abs + param[1]*ty tx dSlope_fringe + param[2]*ty
      // dSlope_fringe^2 + param[3]*ty^3 dSlope_fringe_abs + param[4]*ty tx^2
      // dSlope_fringe_abs
      static constexpr std::array<float, 5> cyParams{ -1.2034772990836242e-05f, 8.344645618037317e-05f,
                                                      -3.924972865228243e-05f, 0.00024639290417116324f,
                                                      0.0001867723161873795f };
    };

    namespace ModSciFiHits {

      namespace HitTag {

        struct fulldex : Event::int_field {};
        struct coord : Event::float_field {};

        template <typename T>
        using hit_t = Event::SOACollection<T, fulldex, coord>;
      } // namespace HitTag

      /**
       * @brief Internal container for SciFi hits in SoA layout
       * @details The container can hold a coordinate and an index. It's used in the PrForwardTracking to keep a
       * subset of (modified) information provided by FT::Hits. It also stores range indices for candidates.
       */
      struct ModPrSciFiHitsSOA : HitTag::hit_t<ModPrSciFiHitsSOA> {

        using base_t = typename HitTag::hit_t<ModPrSciFiHitsSOA>;
        using base_t::base_t;

        void init( size_t sizeHits, size_t sizeCand ) {
          reserve( sizeHits );
          candidateStartIndex.reserve( sizeCand );
          candidateEndIndex.reserve( sizeCand );
        }

        void clear() {
          base_t::clear();
          candidateStartIndex.clear();
          candidateEndIndex.clear();
        }

        auto coord( int index ) const { return *std::next( data<HitTag::coord>(), index ); }
        auto fulldex( int index ) const { return *std::next( data<HitTag::fulldex>(), index ); }

        std::vector<int> candidateStartIndex{};
        std::vector<int> candidateEndIndex{};
      };

    } // namespace ModSciFiHits

    /**
     * @class XCandidate PrForwardTrack.h
     * @brief It's almost a track but contains a lot of temporary quantities used for building a candidate.
     * @details Keeps track of hits and their planes, allows for more than one hit per plane. Provides a simple
     * straight line fit of positions on the reference plane.
     */
    class XCandidate {
    public:
      // typically we look at 3 bins from the HoughTransformation each can have 16 x hits at most
      // so let's take up to 6 hits per xlayer
      static constexpr int planeMulti{ 6 };

      XCandidate( const ModSciFiHits::ModPrSciFiHitsSOA& allXHits, const FT::Hits& SciFiHits )
          : m_allXHits{ allXHits }, m_SciFiHits{ SciFiHits } {}

      void tryToShrinkRange( int, int, float, int );

      auto  nSinglePlanes() const { return std::count( m_planeSize.begin(), m_planeSize.end(), 1 ); }
      auto  nDifferentPlanes() const { return m_nDifferentPlanes; }
      auto& nDifferentPlanes() { return m_nDifferentPlanes; }
      auto& getCoordsToFit() { return m_coordsToFit; }
      auto  getCoordsToFit() const { return m_coordsToFit; }
      auto& getXParams() { return m_xParams; }
      auto  getXParams() const { return m_xParams; }
      auto& getYParams() { return m_yParams; }
      auto  getYParams() const { return m_yParams; }
      auto  nInSamePlane( int fulldex ) const { return m_planeSize[m_SciFiHits.planeCode( fulldex ) / 2u]; }
      auto  planeSize( int ip ) const { return m_planeSize[ip]; }
      auto& planeSize( int ip ) { return m_planeSize[ip]; }
      void  setPlaneSize( int ip, int size ) { m_planeSize[ip] = size; }
      void  setChi2NDoF( std::pair<float, float>&& chi2NDoF ) { m_chi2NDoF = chi2NDoF; }
      auto  getChi2NDoF() const { return m_chi2NDoF; }
      auto  chi2PerDoF() const {
        assert( m_chi2NDoF.second > 0.f );
        return m_chi2NDoF.first / m_chi2NDoF.second;
      }
      void removeFromPlane( int fulldex ) {
        m_nDifferentPlanes -= !( --m_planeSize[m_SciFiHits.planeCode( fulldex ) / 2u] );
      }
      void  removePlane( int ip ) { m_nDifferentPlanes -= !( --m_planeSize[ip] ); }
      auto  xAtRef() const { return m_xAtRef; }
      auto& xAtRef() { return m_xAtRef; }
      auto  txNew() const { return m_txNew; }
      auto  getIdx( int index ) const { return m_idxOnPlane[index]; }
      auto  getIdxSpan( int ip ) const {
        const auto span = LHCb::span{ m_idxOnPlane };
        return span.subspan( ip * planeMulti, m_planeSize[ip] );
      }
      template <auto N>
      void addXParams( LHCb::span<const float, N> pars ) {
        m_xParams.add( pars );
      }

      void clear() {
        m_planeSize.fill( 0 );
        m_nDifferentPlanes = 0;
        m_s.fill( 0.f );
        m_coordsToFit.clear();
        m_xAtRef = 0.f;
      }

      [[gnu::always_inline]] void addHit( int idx ) {
        const int ip = m_SciFiHits.planeCode( m_allXHits.fulldex( idx ) ) / 2;
        if ( const auto pcSize = m_planeSize[ip]; pcSize < planeMulti ) {
          m_idxOnPlane[ip * planeMulti + pcSize] = idx;
          m_nDifferentPlanes += !pcSize;
          ++m_planeSize[ip];
        }
      }

      void addHitForLineFit( int idx, const VeloSeedExtended& veloSeed ) {
        const auto fulldex = m_allXHits.fulldex( idx );
        const auto c       = m_allXHits.coord( idx );
        const auto w       = m_SciFiHits.w( fulldex );
        const auto betterZ = veloSeed.betterZ[m_SciFiHits.planeCode( fulldex )];
        const auto dz      = betterZ - zReference;
        m_s[0] += w;
        m_s[1] += w * dz;
        m_s[2] += w * dz * dz;
        m_s[3] += w * c;
        m_s[4] += w * c * dz;
        m_coordsToFit.push_back( fulldex );
      }

      void solveLineFit() {
        // sz * sz - s0 * sz2
        const auto den = m_s[1] * m_s[1] - m_s[0] * m_s[2];
        // scz * sz - sc * sz2
        m_xAtRef = ( m_s[4] * m_s[1] - m_s[3] * m_s[2] ) / den;
        // sc * sz - s0 * scz
        m_txNew = ( m_s[3] * m_s[1] - m_s[0] * m_s[4] ) / den;
      }

      auto lineChi2( int idx, const VeloSeedExtended& veloSeed ) const {
        const auto fulldex = m_allXHits.fulldex( idx );
        const auto c       = m_allXHits.coord( idx );
        const auto z       = veloSeed.betterZ[m_SciFiHits.planeCode( fulldex )];
        const auto d       = ( c - m_xAtRef ) - ( z - zReference ) * m_txNew;
        return d * d * m_SciFiHits.w( fulldex );
      }

      auto yStraight( float z ) const { return m_yParams[0] + ( z - zReference ) * m_yParams[1]; }

      auto xSlope( float dz ) const { return m_xParams[1] + dz * ( 2.f * m_xParams[2] + 3.f * dz * m_xParams[3] ); }

      auto x( float dz ) const {
        assert( dz < 4000.f && "need distance to reference plane here!" );
        return m_xParams[0] + dz * ( m_xParams[1] + dz * ( m_xParams[2] + dz * m_xParams[3] ) );
      }

      auto y( float dz ) const {
        assert( dz < 4000.f && "need distance to reference plane here!" );
        return m_yParams[0] + dz * ( m_yParams[1] + dz * m_yParams[2] );
      }

      auto calcBetterDz( int fulldex ) const {
        const auto dz = m_SciFiHits.z( fulldex ) - zReference;
        return dz + m_SciFiHits.dzDy( fulldex ) * y( dz );
      }

      auto distanceXHit( int fulldex, float dz ) const { return m_SciFiHits.x( fulldex ) - x( dz ); }

      auto chi2XHit( int fulldex, float dz ) const {
        const auto d = distanceXHit( fulldex, dz );
        return d * d * m_SciFiHits.w( fulldex );
      }

      auto planeEmpty( int idx ) const {
        const int ip = m_SciFiHits.planeCode( m_allXHits.fulldex( idx ) ) / 2;
        return !m_planeSize[ip];
      }

      auto improveRightSide( int idx1, int idx2, int idxEnd, float maxXGap, float xWindow ) {
        for ( auto idxLast = idx2 - 1; idx2 < idxEnd; ++idx2 ) {
          direct_debug( "Gap to the right of cluster =", m_allXHits.coord( idx2 ) - m_allXHits.coord( idxLast ), "<",
                        "maxXGap =", maxXGap );
          const auto smallGap = m_allXHits.coord( idx2 ) < m_allXHits.coord( idxLast ) + maxXGap;
          const auto inWindow = m_allXHits.coord( idx2 ) - m_allXHits.coord( idx1 ) < xWindow;
          if ( !( smallGap || ( inWindow && planeEmpty( idx2 ) ) ) ) return idx2;
          addHit( idx2 );
          idxLast = idx2;
        }
        return idx2;
      }

    private:
      alignas( 64 ) std::array<int, Detector::FT::nXLayersTotal * planeMulti> m_idxOnPlane{};
      alignas( 64 ) std::array<int, Detector::FT::nXLayersTotal> m_planeSize{};
      alignas( 64 ) std::array<float, 5> m_s{ nan, nan, nan, nan, nan };
      FitCoords<int, Detector::FT::nXLayersTotal * planeMulti> m_coordsToFit{};

      // unsigned to avoid warnings
      unsigned                               m_nDifferentPlanes{ 0 };
      float                                  m_txNew{ nan };
      float                                  m_xAtRef{ nan };
      TrackPars<4>                           m_xParams{ { nan, nan, nan, nan } };
      TrackPars<3>                           m_yParams{ { nan, nan, nan } };
      std::pair<float, float>                m_chi2NDoF{ nan, nan };
      const ModSciFiHits::ModPrSciFiHitsSOA& m_allXHits;
      const FT::Hits&                        m_SciFiHits;

      // for debugging
      friend std::ostream& operator<<( std::ostream& os, const XCandidate& t ) {
        return os << "XCandidate: "
                  << "xPars = " << t.m_xParams << ", "
                  << "yPars = " << t.m_yParams << ", "
                  << "(chi2, nDoF) = "
                  << "( " << t.m_chi2NDoF.first << ", " << t.m_chi2NDoF.second << " )" << std::endl
                  << "hits = " << t.m_coordsToFit;
      }
    };

    inline void XCandidate::tryToShrinkRange( int idx1, int idx2, float maxInterval, int nPlanes ) {
      auto idxWindowStart = idx1;
      auto idxWindowEnd   = idx1 + nPlanes; // pointing at last+1

      auto best    = idx1;
      auto bestEnd = idx2;

      auto       otherNDifferentPlanes{ 0 };
      std::array otherPlaneSize{ 0, 0, 0, 0, 0, 0 };

      for ( auto idx{ idxWindowStart }; idx < idxWindowEnd; ++idx ) {
        const int ip = m_SciFiHits.planeCode( m_allXHits.fulldex( idx ) ) / 2;
        otherNDifferentPlanes += !( otherPlaneSize[ip]++ );
      }

      while ( idxWindowEnd <= idx2 ) {
        if ( otherNDifferentPlanes >= nPlanes ) {
          // have nPlanes, check x distance
          const float dist = m_allXHits.coord( idxWindowEnd - 1 ) - m_allXHits.coord( idxWindowStart );
          direct_debug( "current subrange width =", dist, "< best width", maxInterval );
          if ( dist < maxInterval ) {
            maxInterval = dist;
            best        = idxWindowStart;
            bestEnd     = idxWindowEnd;
            // need to keep track of planes
            m_nDifferentPlanes = otherNDifferentPlanes;
            m_planeSize        = otherPlaneSize;
          }
        } else {
          // too few planes, add one hit
          ++idxWindowEnd;
          if ( idxWindowEnd > idx2 ) break;
          const int ip = m_SciFiHits.planeCode( m_allXHits.fulldex( idxWindowEnd - 1 ) ) / 2;
          otherNDifferentPlanes += !( otherPlaneSize[ip]++ );
          continue;
        }
        // move on to the right
        const int ip = m_SciFiHits.planeCode( m_allXHits.fulldex( idxWindowStart ) ) / 2;
        otherNDifferentPlanes -= !( --otherPlaneSize[ip] );
        ++idxWindowStart;
      }
      for ( auto idx{ best }; idx < bestEnd; ++idx ) {
        const auto fulldex = m_allXHits.fulldex( idx );
        m_coordsToFit.push_back( fulldex );
        m_xAtRef += m_allXHits.coord( idx );
      }
      m_xAtRef /= static_cast<float>( m_coordsToFit.size() );
    }

    /**
     * @brief A class keeping information about the stereo candidate necessary for fitting.
     *
     * @details It is mostly used for fitting and thus provides functions needed by the fits.
     * It also keeps track of planes that are already used by the candidate -- it can only hold
     * one hit per layer.
     */
    struct StereoCandidate {
    public:
      auto& getXParams() { return m_xParams; }
      auto  getXParams() const { return m_xParams; }
      auto& getYParams() { return m_yParams; }
      auto  getYParams() const { return m_yParams; }
      auto& getCoordsToFit() { return m_coordsToFit; }
      auto  getCoordsToFit() const { return m_coordsToFit; }
      auto& getUsedPlanes() { return m_usedPlanes; }
      auto  nDifferentPlanes() const { return m_usedPlanes.count(); }
      void  setPlaneUsed( int ip ) noexcept { m_usedPlanes[ip] = true; }
      auto  isPlaneUsed( int ip ) const noexcept { return m_usedPlanes[ip]; }

      template <auto N>
      void addYParams( LHCb::span<const float, N> pars ) {
        m_yParams.add( pars );
      }

      void clear() {
        m_usedPlanes.reset();
        m_coordsToFit.clear();
      }

      auto x( float dz ) const {
        assert( dz < 4000.f && "need distance to reference plane here!" );
        return m_xParams[0] + dz * ( m_xParams[1] + dz * ( m_xParams[2] + dz * m_xParams[3] ) );
      }

      auto y( float dz ) const {
        assert( dz < 4000.f && "need distance to reference plane here!" );
        return m_yParams[0] + dz * ( m_yParams[1] + dz * m_yParams[2] );
      }

      auto yStraight( float z ) const { return m_yParams[0] + ( z - zReference ) * m_yParams[1]; }

      auto xSlope( float dz ) const { return m_xParams[1] + dz * ( 2.f * m_xParams[2] + 3.f * dz * m_xParams[3] ); }

      auto calcBetterDz( int fulldex, const FT::Hits& SciFiHits ) const {
        const auto dz = SciFiHits.z( fulldex ) - zReference;
        return dz + SciFiHits.dzDy( fulldex ) * y( dz );
      }

      auto distanceStereoHit( int fulldex, float dz, const FT::Hits& SciFiHits ) const {
        return ( x( dz ) - ( SciFiHits.x( fulldex ) + y( dz ) * SciFiHits.dxDy( fulldex ) ) ) /
               SciFiHits.dxDy( fulldex );
      }

      auto chi2StereoHits( int fulldex, const FT::Hits& SciFiHits ) const {
        const auto dz = calcBetterDz( fulldex, SciFiHits );
        const auto d  = ( SciFiHits.x( fulldex ) + y( dz ) * SciFiHits.dxDy( fulldex ) ) - x( dz );
        return d * d * SciFiHits.w( fulldex );
      }

      void removeFromPlane( int fulldex, const FT::Hits& SciFiHits ) noexcept {
        m_usedPlanes[SciFiHits.planeCode( fulldex ) / 2u] = false;
      }

    private:
      FitCoords<int, Detector::FT::nUVLayersTotal> m_coordsToFit{};
      TrackPars<4>                                 m_xParams{ { nan, nan, nan, nan } };
      TrackPars<3>                                 m_yParams{ { nan, nan, nan } };
      std::bitset<Detector::FT::nUVLayersTotal>    m_usedPlanes{};

      // for debugging
      friend std::ostream& operator<<( std::ostream& os, const StereoCandidate& t ) {
        return os << "StereoCandidate: "
                  << "xPars = " << t.m_xParams << ", "
                  << "yPars = " << t.m_yParams << ", "
                  << "usedPlanes = " << t.m_usedPlanes << std::endl
                  << "hits = " << t.m_coordsToFit;
      }
    };

    /**
     * @class PrForwardTrack PrForwardTrack.h
     * @brief Internal representation of a long track.
     * @details Only one hit per plane is allowed, has all information needed to make a final long track
     * out of it, i.e. knows its Velo(UT) track and q/p, and the quantities needed to obtain it.
     */
    class PrForwardTrack {
    public:
      PrForwardTrack( static_vector<int, Detector::FT::nXLayersTotal * XCandidate::planeMulti> coordsToFit,
                      TrackPars<4> xParams, TrackPars<3> yParams, std::pair<float, float> chi2NDoF, int iInputTrack )
          : m_xParams{ xParams }, m_yParams{ yParams }, m_chi2NDoF{ chi2NDoF }, m_track{ iInputTrack } {
        std::copy( std::make_move_iterator( coordsToFit.begin() ), std::make_move_iterator( coordsToFit.end() ),
                   std::back_inserter( m_coordsToFit ) );
      }

      int   track() const { return m_track; }
      auto  size() const { return m_coordsToFit.size(); }
      auto  getXParams() const { return m_xParams; }
      auto& getXParams() { return m_xParams; }
      auto  getYParams() const { return m_yParams; }
      auto& getYParams() { return m_yParams; }

      auto&       getCoordsToFit() { return m_coordsToFit; }
      const auto& getCoordsToFit() const { return m_coordsToFit; }
      auto        valid() const { return m_valid; }
      void        setValid() { m_valid = true; }
      void        setInvalid() { m_valid = false; }

      void setQuality( float q ) { m_quality = q; }
      auto quality() const { return m_quality; }

      void setQoP( float qop ) { m_qop = qop; }
      auto getQoP() const { return m_qop; }

      template <typename Container>
      void addHits( Container&& hits ) {
        std::copy( std::make_move_iterator( hits.begin() ), std::make_move_iterator( hits.end() ),
                   std::back_inserter( m_coordsToFit ) );
      }

      auto x( float dz ) const {
        assert( dz < 4000.f && "need distance to reference plane here!" );
        return m_xParams[0] + dz * ( m_xParams[1] + dz * ( m_xParams[2] + dz * m_xParams[3] ) );
      }

      auto y( float dz ) const {
        assert( dz < 4000.f && "need distance to reference plane here!" );
        return m_yParams[0] + dz * ( m_yParams[1] + dz * m_yParams[2] );
      }

      auto distance( int fulldex, float dz, const FT::Hits& SciFiHits ) const {
        return ( SciFiHits.x( fulldex ) + y( dz ) * SciFiHits.dxDy( fulldex ) ) - x( dz );
      }

      auto chi2( int fulldex, float dz, const FT::Hits& SciFiHits ) const {
        const auto d = distance( fulldex, dz, SciFiHits );
        return d * d * SciFiHits.w( fulldex );
      }

      auto getBetterDz( int fulldex, float dz, const FT::Hits& SciFiHits ) {
        dz += y( dz ) * SciFiHits.dzDy( fulldex );
        return dz;
      }

      auto xSlope( float dz ) const { return m_xParams[1] + dz * ( 2.f * m_xParams[2] + 3.f * dz * m_xParams[3] ); }

      auto ySlope( float dz ) const { return m_yParams[1] + dz * 2.f * m_yParams[2]; }

      template <auto N>
      void addXParams( LHCb::span<const float, N> pars ) {
        m_xParams.add( pars );
      }

      void setChi2NDoF( std::pair<float, float>&& chi2NDoF ) { m_chi2NDoF = chi2NDoF; }
      auto getChi2PerNDoF() const { return m_chi2NDoF.first / m_chi2NDoF.second; }

      /**
       * @brief Estimates charge over momentum for the Forward Track.
       *
       * @param veloSeed Velo track information (here only the slopes are needed).
       * @param magScaleFactor Relative signed current of the magnet.
       * @return Estimate of charge over momentum in MeV.
       *
       * @details This uses a parameterisation obtained using the Reco-Parameterisation-Tuner.
       * https://gitlab.cern.ch/gunther/prforwardtracking-parametrisation-tuner/-/tree/master
       */
      auto estimateChargeOverMomentum( const VeloSeedExtended& veloSeed, float magScaleFactor ) {
        const auto tx        = veloSeed.seed.tx;
        const auto tx2       = veloSeed.seed.tx2;
        const auto tx4       = tx2 * tx2;
        const auto ty2       = veloSeed.seed.ty2;
        const auto tx_ref    = m_xParams[1];
        const auto tx_ref2   = tx_ref * tx_ref;
        const auto tx_ref4   = tx_ref2 * tx_ref2;
        const auto dSlopeMag = std::abs( magScaleFactor ) > 0.f ? ( tx_ref - tx ) / magScaleFactor : nanMomentum;
        const auto integral =
            fieldIntegralParamsRef[0] +
            ty2 * ( fieldIntegralParamsRef[1] + fieldIntegralParamsRef[5] * ty2 + fieldIntegralParamsRef[6] * tx2 ) +
            tx * tx_ref *
                ( fieldIntegralParamsRef[10] * tx2 + fieldIntegralParamsRef[3] + fieldIntegralParamsRef[7] * ty2 ) +
            fieldIntegralParamsRef[11] * tx_ref4 + fieldIntegralParamsRef[2] * tx2 +
            tx_ref2 * ( fieldIntegralParamsRef[4] + fieldIntegralParamsRef[8] * ty2 ) + fieldIntegralParamsRef[9] * tx4;
        m_qop = dSlopeMag / ( integral * float{ Gaudi::Units::GeV } );
        return m_qop;
      }

    private:
      FitCoords<int, Detector::FT::nLayersTotal> m_coordsToFit{};
      TrackPars<4>                               m_xParams{ { nan, nan, nan, nan } };
      TrackPars<3>                               m_yParams{ { nan, nan, nan } };
      std::pair<float, float>                    m_chi2NDoF{ nan, nan };
      float                                      m_quality{ 0.f };
      float                                      m_qop{ nan };
      int                                        m_track{ -1 };
      bool                                       m_valid{ false };

      // for debugging
      friend std::ostream& operator<<( std::ostream& os, const PrForwardTrack& t ) {
        return os << "PrForwardTrack for input " << t.m_track << " :"
                  << "xPars = " << t.m_xParams << ", "
                  << "yPars = " << t.m_yParams << ", "
                  << "(chi2, nDoF) = "
                  << "( " << t.m_chi2NDoF.first << ", " << t.m_chi2NDoF.second << " )"
                  << ", "
                  << "quality = " << t.m_quality << ", "
                  << "qop = " << t.m_qop << ", "
                  << "valid = " << t.m_valid << std::endl
                  << "hits = " << t.m_coordsToFit;
      }

      // param[0] + param[1]*ty^2 + param[2]*tx^2 + param[3]*tx tx_ref +
      // param[4]*tx_ref^2 + param[5]*ty^4 + param[6]*ty^2 tx^2 + param[7]*ty^2 tx
      // tx_ref + param[8]*ty^2 tx_ref^2 + param[9]*tx^4 + param[10]*tx^3 tx_ref +
      // param[11]*tx_ref^4
      static constexpr std::array<float, 12> fieldIntegralParamsRef{
          -1.2094486121528516f, -2.7897043324822492f, -0.35976930628193077f, -0.47138558705675454f,
          -0.5600847231491961f, 14.009315350693472f,  -16.162818973243674f,  -8.807994419844437f,
          -0.8753190393972976f, 2.98254201374128f,    0.9625408279466898f,   0.10200564097830103f };
    };
  } // namespace Forward
} // namespace LHCb::Pr
