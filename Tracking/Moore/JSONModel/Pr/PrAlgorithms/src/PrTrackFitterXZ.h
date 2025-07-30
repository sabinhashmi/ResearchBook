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

#ifndef PRTRACKFITTERXZ_H
#define PRTRACKFITTERXZ_H 1

// Include files
#include "FTDAQ/FTInfo.h"
#include "LHCbMath/MatVec.h"
#include "PrHybridSeedTrack.h"
#include <array>

/** @class PrTrackFitterXZ PrTrackFitterXZ.h
 *  Support class to perform fit of x-z track projections
 *  @author Renato Quagliani
 *  @date   2015-08-01
 *  @author Salvatore Aiola (salvatore.aiola@cern.ch)
 *  @date   2020-02-29
 */

class PrTrackFitterXZ final {
  static constexpr size_t N_MAX_HITS     = LHCb::Detector::FT::nXLayersTotal;
  static constexpr float  ZREF           = LHCb::Pr::Hybrid::zReference;
  static constexpr int    N_FIT_PARAMS   = 3;
  static constexpr float  LINEAR_SCALING = 1e-3f; // Scaling factor used to improve numerical stability
  static constexpr float  CUBIC_SCALING =
      LINEAR_SCALING * LINEAR_SCALING * LINEAR_SCALING; // Scaling factor used to improve numerical stability

  using TrackType    = LHCb::Pr::Hybrid::SeedTrackX;
  using TrackTypeRef = TrackType&;
  using Solver       = ROOT::Math::CholeskyDecomp<float, N_FIT_PARAMS>;
  using MatrixType   = LHCb::LinAlg::MatSym<float, N_FIT_PARAMS>;
  using VectorType   = LHCb::LinAlg::Vec<float, N_FIT_PARAMS>;
  using PrSciFiHits  = LHCb::Pr::FT::Hits;

public:
  PrTrackFitterXZ()                                    = default;
  PrTrackFitterXZ( const PrTrackFitterXZ& )            = delete;
  PrTrackFitterXZ( PrTrackFitterXZ&& )                 = delete;
  PrTrackFitterXZ& operator=( const PrTrackFitterXZ& ) = delete;
  PrTrackFitterXZ& operator=( PrTrackFitterXZ&& )      = delete;

  bool fit( const PrSciFiHits& sciFiHits, TrackType& track ) noexcept {
    VectorType rhs; // right-hand-side of the linear system (function of the x positions)
    MatrixType mat; // matrix of the linear system

    initialize_linear_system( sciFiHits, track, mat, rhs );

    // Decompose matrix
    Solver decomp( mat.m.data() ); //---LoH: can probably be made more rapidly
    if ( !decomp ) return false;

    // Solve linear system
    VectorType result( rhs );
    decomp.Solve( result.m );
    result( 1 ) *= LINEAR_SCALING; // rescale back the fit parameters (dz rescaled for numerical stability)
    result( 2 ) *= CUBIC_SCALING;  // rescale back the fit parameters (dz rescaled for numerical stability)

    // Small corrections
    track.updateParameters( result( 0 ), result( 1 ), result( 2 ) );

    calculate_chi2( sciFiHits, track );

    return true;
  }

private:
  void calculate_chi2( const PrSciFiHits& sciFiHits, TrackType& track ) noexcept {
    // Calculate chi2
    float chi2PerDoF = 0.f;
    for ( auto [iterH, iterChi2] = std::tuple{ track.hits().cbegin(), track.chi2Hits().begin() };
          iterH != track.hits().cend(); ++iterH, ++iterChi2 ) {
      *iterChi2 = track.chi2( sciFiHits, *( iterH ) );
      chi2PerDoF += *iterChi2;
    }
    chi2PerDoF /= ( track.hits().size() - N_FIT_PARAMS );
    track.setChi2PerDoF( chi2PerDoF );
  }

  void initialize_linear_system( const PrSciFiHits& sciFiHits, TrackType& track, MatrixType& mat,
                                 VectorType& rhs ) noexcept {
    mat.m.fill( 0.f );
    rhs.m.fill( 0.f );
    const float dRatio = track.dRatio();
    for ( const auto& modHit : track.hits() ) {
      const auto  i     = modHit.fullDex;
      const float w     = sciFiHits.w( i ); // squared
      const float dzpr  = ( sciFiHits.z( i ) - ZREF );
      const float dz    = LINEAR_SCALING * dzpr; // rescale dz for numerical stability
      const float wdz   = w * dz;
      const float deta  = dz * dz * ( LINEAR_SCALING + dRatio * dz ); // rescale dz for numerical stability
      const float wdeta = w * deta;
      const float dist  = track.distanceFromDz( modHit, dzpr );
      mat( 0, 0 ) += w;
      mat( 1, 0 ) += wdz;
      mat( 1, 1 ) += wdz * dz;
      mat( 2, 0 ) += wdeta;
      mat( 2, 1 ) += wdeta * dz;
      mat( 2, 2 ) += wdeta * deta;
      rhs( 0 ) += w * dist;
      rhs( 1 ) += wdz * dist;
      rhs( 2 ) += wdeta * dist;
    }
  }
};

#endif // PRTRACKFITTERX_H
