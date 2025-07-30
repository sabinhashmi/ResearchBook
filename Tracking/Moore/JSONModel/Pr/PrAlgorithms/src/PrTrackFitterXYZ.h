/*******************************************************************************\
 * (c) Copyright 2000-2020 CERN for the benefit of the LHCb Collaboration      *
 *                                                                             *
 * This software is distributed under the terms of the GNU General Public      *
 * Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
\*******************************************************************************/

#ifndef PRTRACKFITTERXYZ_H
#define PRTRACKFITTERXYZ_H 1

// Include files
#include "FTDAQ/FTInfo.h"
#include "LHCbMath/MatVec.h"
#include "Math/CholeskyDecomp.h"
#include "PrHybridSeedTrack.h"
#include <array>

/** @class PrTrackFitterXYZ PrTrackFitterXYZ.h
 *  Support class to perform fit of 3D tracks
 *  @author Renato Quagliani
 *  @date   2015-08-01
 *  @author Salvatore Aiola (salvatore.aiola@cern.ch)
 *  @date   2020-02-29
 */
template <int NITER>
class PrTrackFitterXYZ final {
  static constexpr size_t N_MAX_HITS     = LHCb::Detector::FT::nLayersTotal;
  static constexpr float  ZREF           = LHCb::Pr::Hybrid::zReference;
  static constexpr int    N_FIT_PARAMS   = 5;
  static constexpr float  LINEAR_SCALING = 1e-3f; // Scaling factor used to improve numerical stability
  static constexpr float  CUBIC_SCALING =
      LINEAR_SCALING * LINEAR_SCALING * LINEAR_SCALING; // Scaling factor used to improve numerical stability

  using TrackType    = LHCb::Pr::Hybrid::SeedTrack;
  using TrackTypeRef = TrackType&;
  using Solver       = ROOT::Math::CholeskyDecomp<float, N_FIT_PARAMS>;
  using MatrixType   = LHCb::LinAlg::MatSym<float, N_FIT_PARAMS>;
  using VectorType   = LHCb::LinAlg::Vec<float, N_FIT_PARAMS>;
  using PrSciFiHits  = LHCb::Pr::FT::Hits;

public:
  PrTrackFitterXYZ()                                            = default;
  PrTrackFitterXYZ( const PrTrackFitterXYZ<NITER>& )            = delete;
  PrTrackFitterXYZ( PrTrackFitterXYZ<NITER>&& )                 = delete;
  PrTrackFitterXYZ& operator=( const PrTrackFitterXYZ<NITER>& ) = delete;
  PrTrackFitterXYZ& operator=( PrTrackFitterXYZ<NITER>&& )      = delete;

  template <typename DRatioParArrayType>
  bool fit( const PrSciFiHits& sciFiHits, TrackType& track, const DRatioParArrayType& dRatioPar ) noexcept {
    VectorType rhs; // right-hand-side of the linear system (function of the x positions)
    MatrixType mat; // matrix of the linear system

    initialize_matrix( sciFiHits, track, mat );

    std::optional<Solver> decomp;

    for ( int loop = 0; loop < NITER; ++loop ) {
      if ( loop == 1 ) {
        // See: https://cds.cern.ch/record/2027531, Fig. 9
        float radius2   = ( track.ax() * track.ax() / 4 + track.yRef() * track.yRef() ) * 1e-6f;
        float radius    = std::sqrt( radius2 );
        float dRatioPos = -( dRatioPar[0] + dRatioPar[1] * radius + dRatioPar[2] * radius2 );
        track.setdRatio( dRatioPos );
      }

      initialize_linear_system( sciFiHits, track, mat, rhs );

      // Decompose matrix, protect against numerical trouble
      decomp.emplace( mat.m.data() );
      if ( !decomp->ok() ) return false;

      // Solve linear system
      VectorType result( rhs );
      decomp->Solve( result.m );
      result( 4 ) *= LINEAR_SCALING; // rescale back the fit parameters (dz rescaled for numerical stability)
      result( 3 ) -= result( 4 ) * ZREF;
      result( 2 ) *= CUBIC_SCALING;  // rescale back the fit parameters (dz rescaled for numerical stability)
      result( 1 ) *= LINEAR_SCALING; // rescale back the fit parameters (dz rescaled for numerical stability)

      // Apply results to track object
      track.updateParameters( result( 0 ), result( 1 ), result( 2 ), result( 3 ), result( 4 ) );
    }

    calculate_chi2( sciFiHits, track );

    return true;
  }

private:
  void calculate_chi2( const PrSciFiHits& sciFiHits, TrackType& track ) noexcept {
    // Calculate chi2
    float chi2PerDoF = 0.f;
    for ( auto [iterH, iterChi2] = std::tuple{ track.hits().cbegin(), track.chi2Hits().begin() };
          iterH != track.hits().cend(); ++iterH, ++iterChi2 ) {
      *iterChi2 = track.chi2( sciFiHits, iterH->fullDex );
      chi2PerDoF += *iterChi2;
    }
    chi2PerDoF /= ( track.hits().size() - N_FIT_PARAMS );
    track.setChi2PerDoF( chi2PerDoF );
  }

  void initialize_matrix( const PrSciFiHits& sciFiHits, TrackType& track, MatrixType& mat ) noexcept {
    mat( 0, 0 ) = 0.f;
    mat( 3, 0 ) = 0.f;
    mat( 3, 3 ) = 0.f;
    for ( const auto& modHit : track.hits() ) {
      const int   fullDex = modHit.fullDex;
      const float w       = sciFiHits.w( fullDex );
      const float dxdy    = sciFiHits.dxDy( fullDex );
      const float wdxdy   = w * dxdy;
      mat( 0, 0 ) += w;
      mat( 3, 0 ) -= wdxdy;
      mat( 3, 3 ) += wdxdy * dxdy;
    } // Loop over Hits to fill the matrix
  }

  void initialize_linear_system( const PrSciFiHits& sciFiHits, TrackType& track, MatrixType& mat,
                                 VectorType& rhs ) noexcept {
    rhs.m.fill( 0.f );
    mat( 1, 0 ) = 0.f;
    mat( 1, 1 ) = 0.f;
    mat( 2, 0 ) = 0.f;
    mat( 2, 1 ) = 0.f;
    mat( 2, 2 ) = 0.f;
    mat( 3, 1 ) = 0.f;
    mat( 3, 2 ) = 0.f;
    mat( 4, 0 ) = 0.f;
    mat( 4, 1 ) = 0.f;
    mat( 4, 2 ) = 0.f;
    mat( 4, 3 ) = 0.f;
    mat( 4, 4 ) = 0.f;
    for ( const auto& modHit : track.hits() ) {
      const int   fullDex  = modHit.fullDex;
      const float w        = sciFiHits.w( fullDex ); // squared
      const float dxdy     = sciFiHits.dxDy( fullDex );
      const float yOnTrack = track.yOnTrack( sciFiHits, fullDex );
      const float dz =
          LINEAR_SCALING * ( sciFiHits.z( fullDex, yOnTrack ) - ZREF );         // rescale dz for numerical stability
      const float deta    = dz * dz * ( LINEAR_SCALING + dz * track.dRatio() ); // rescale dz for numerical stability
      const float wdz     = w * dz;
      const float weta    = w * deta;
      const float wdxdy   = w * dxdy;
      const float wdxdydz = wdxdy * dz;
      const float dist    = track.distance( sciFiHits, fullDex );
      mat( 1, 0 ) += wdz;
      mat( 1, 1 ) += wdz * dz;
      mat( 2, 0 ) += weta;
      mat( 2, 1 ) += weta * dz;
      mat( 2, 2 ) += weta * deta;
      mat( 3, 1 ) -= wdxdydz;
      mat( 3, 2 ) -= wdxdy * deta;
      mat( 4, 0 ) -= wdxdydz;
      mat( 4, 1 ) -= wdxdydz * dz;
      mat( 4, 2 ) -= wdxdydz * deta;
      mat( 4, 3 ) += wdxdydz * dxdy;
      mat( 4, 4 ) += wdxdydz * dz * dxdy;
      // fill right hand side
      rhs( 0 ) += w * dist;
      rhs( 1 ) += wdz * dist;
      rhs( 2 ) += weta * dist;
      rhs( 3 ) -= wdxdy * dist;
      rhs( 4 ) -= wdxdydz * dist;
    } // Loop over Hits to fill the matrix
  }
};

#endif // PRTRACKFITTERXYZ_H
