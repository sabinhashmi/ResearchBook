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
#pragma once

#include "Core/FloatComparison.h"
#include "FTDAQ/FTInfo.h"
#include "LHCbMath/MatVec.h"
#include "PrHybridSeedTrack.h"
#include <array>

/** @class PrTrackFitterYZ PrTrackFitterYZ.h
 *  Support class to perform fit of lines using solely hits in UV layers setting the parameters of
 *  the xz projection and fitting passing iterators of the hough-like cluster
 *  @author Renato Quagliani
 *  @date   2015-08-01
 *  @author Salvatore Aiola (salvatore.aiola@cern.ch)
 *  @date   2020-02-29
 */

class PrTrackFitterYZ final {
  static constexpr size_t N_MAX_HITS   = LHCb::Detector::FT::nUVLayersTotal;
  static constexpr float  ZREF         = LHCb::Pr::Hybrid::zReference;
  static constexpr int    N_FIT_PARAMS = 2;

  using TrackType    = const LHCb::Pr::Hybrid::SeedTrackX;
  using TrackTypeRef = TrackType&;
  using MatrixType   = LHCb::LinAlg::MatSym<float, N_FIT_PARAMS>;
  using VectorType   = LHCb::LinAlg::Vec<float, N_FIT_PARAMS>;
  using PrSciFiHits  = LHCb::Pr::FT::Hits;

public:
  PrTrackFitterYZ()                                    = delete;
  PrTrackFitterYZ( const PrTrackFitterYZ& )            = delete;
  PrTrackFitterYZ( PrTrackFitterYZ&& )                 = delete;
  PrTrackFitterYZ& operator=( const PrTrackFitterYZ& ) = delete;
  PrTrackFitterYZ& operator=( PrTrackFitterYZ&& )      = delete;

  PrTrackFitterYZ( TrackType& track ) : m_track( track ) {}

  float       chi2PerDoF() const noexcept { return m_chi2PerDoF; }
  const auto& chi2Hits() const noexcept { return m_chi2Hits; }
  auto&       chi2Hits() noexcept { return m_chi2Hits; }
  float       ay() const noexcept { return m_ay; }
  float       by() const noexcept { return m_by; }

  // Fit hit positions with a straight line using Cramer's rule
  template <typename HitIteratorType>
  bool fit( const PrSciFiHits& sciFiHits, HitIteratorType itBeg, HitIteratorType itEnd ) noexcept {
    VectorType rhs; // right-hand-side of the linear system (function of the x positions)
    MatrixType mat; // matrix of the linear system

    initialize_linear_system( sciFiHits, itBeg, itEnd, mat, rhs );

    // Calculate determinant
    float det = mat( 0, 0 ) * mat( 1, 1 ) - mat( 0, 1 ) * mat( 1, 0 );
    if ( LHCb::essentiallyZero( det ) ) return false;

    // Solve linear system
    const float invDet = 1.f / det;
    m_ay               = ( rhs( 0 ) * mat( 1, 1 ) - mat( 0, 1 ) * rhs( 1 ) ) * invDet;
    m_by               = ( mat( 0, 0 ) * rhs( 1 ) - rhs( 0 ) * mat( 1, 0 ) ) * invDet;

    calculate_chi2( sciFiHits, itBeg, itEnd );

    return true;
  }

private:
  template <typename HitIteratorType>
  void calculate_chi2( const PrSciFiHits& sciFiHits, HitIteratorType itBeg, HitIteratorType itEnd ) noexcept {
    // Calculate chi2
    m_chi2PerDoF = 0.f;
    for ( auto [iterH, iterChi2] = std::tuple{ itBeg, m_chi2Hits.begin() }; itEnd != iterH; ++iterH, ++iterChi2 ) {
      *iterChi2 = chi2hit( sciFiHits, **iterH );
      m_chi2PerDoF += *iterChi2;
    }
    m_chi2PerDoF /= ( itEnd - itBeg - N_FIT_PARAMS );
  }

  template <typename HitIteratorType>
  void initialize_linear_system( const PrSciFiHits& sciFiHits, HitIteratorType itBeg, HitIteratorType itEnd,
                                 MatrixType& mat, VectorType& rhs ) noexcept {
    mat.m.fill( 0.f );
    rhs.m.fill( 0.f );
    for ( auto iterH = itBeg; itEnd != iterH; ++iterH ) {
      const ModPrHit& hit    = *( *iterH );
      const auto      i      = ( *iterH )->fullDex;
      const float     dz     = sciFiHits.z( i ) - ZREF;
      const float     dist   = distanceAt0FromDz( hit, dz );
      const float     dxDy   = sciFiHits.dxDy( i );
      const float     wdxDy  = sciFiHits.w( i ) * dxDy;
      const float     wdxDy2 = wdxDy * dxDy;
      mat( 0, 0 ) += wdxDy2;
      mat( 0, 1 ) += wdxDy2 * dz;
      mat( 1, 1 ) += wdxDy2 * dz * dz;
      rhs( 0 ) -= wdxDy * dist;
      rhs( 1 ) -= wdxDy * dz * dist;
    }
  }

  float y( float z ) const noexcept { return ( m_ay + m_by * ( z - ZREF ) ); }
  float yFromDz( float dz ) const noexcept { return ( m_ay + m_by * dz ); }

  float chi2hit( const PrSciFiHits& sciFiHits, const ModPrHit& hit ) const noexcept {
    int         i    = hit.fullDex;
    float       erry = sciFiHits.w( i );
    const float dist = distance( sciFiHits, i );
    return dist * dist * erry;
  }

  float distance( const PrSciFiHits& sciFiHits, int fullDex ) const noexcept {
    const float dz   = sciFiHits.z( fullDex ) - ZREF;
    const float yAtZ = yFromDz( dz );
    return ( sciFiHits.x( fullDex, yAtZ ) - m_track.xFromDz( dz ) );
  }

  float distanceAt0( const ModPrHit& hit, float z ) const noexcept { return ( hit.coord - m_track.x( z ) ); }
  float distanceAt0FromDz( const ModPrHit& hit, float dz ) const noexcept {
    return ( hit.coord - m_track.xFromDz( dz ) );
  }

private:
  TrackTypeRef                  m_track;        // reference to the track candidate
  float                         m_chi2PerDoF{}; // chi2 per degrees of freedom
  float                         m_ay{};         // fit param ay
  float                         m_by{};         // fit param by
  std::array<float, N_MAX_HITS> m_chi2Hits{};   // chi2 contributions of each hit
};
