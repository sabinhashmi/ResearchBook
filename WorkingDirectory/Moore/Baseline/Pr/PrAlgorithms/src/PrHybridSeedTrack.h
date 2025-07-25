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
#pragma once

// Include files
#include "PrKernel/PrHit.h"

#include "Event/PrHits.h"
#include "Event/StateParameters.h"
#include "FTDAQ/FTInfo.h"
#include "LHCbMath/BloomFilter.h"

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

/** @class PrHybridSeedTrack PrHybridSeedTrack.h
 *  This is the working class inside the T station pattern
 *  ---LoH: right now, the x projection is with respect to zRef
 *
 *  @author Renato Quagliani
 *  @author Louis Henry
 *  @date   2020-05-21
 */
// Comment it if you don't want to do truth matching

namespace LHCb::Pr::Hybrid {
  constexpr static float zReference    = StateParameters::ZMidT;
  constexpr unsigned int maxStereoHits = 100;

  using SeedTrackHitsX          = boost::container::static_vector<ModPrHit, LHCb::Detector::FT::nXLayersTotal>;
  using SeedTrackHitsXIter      = SeedTrackHitsX::iterator;
  using SeedTrackHitsXConstIter = SeedTrackHitsX::const_iterator;
  using SeedTrackHits           = boost::container::static_vector<ModPrHit, LHCb::Detector::FT::nLayersTotal>;
  using SeedTrackHitsIter       = SeedTrackHits::iterator;
  using SeedTrackHitsConstIter  = SeedTrackHits::const_iterator;
  using StereoHits              = boost::container::small_vector<ModPrHit, maxStereoHits>;
  using StereoIter              = boost::container::small_vector<ModPrHit, maxStereoHits>::const_iterator;
  using LHCb::Pr::FT::Hits;

  class AbsSeedTrack {
  public:
    AbsSeedTrack()                                  = delete;
    AbsSeedTrack( const AbsSeedTrack& )             = default;
    AbsSeedTrack( const AbsSeedTrack&& )            = delete;
    AbsSeedTrack& operator=( const AbsSeedTrack& )  = default;
    AbsSeedTrack& operator=( const AbsSeedTrack&& ) = delete;
    AbsSeedTrack( float dratio ) : m_dRatio( dratio ) {}

    // Zone (0,1) Up Down
    //====================================
    // SETTERS & Getters  Genearl One
    //====================================
    //***********Track Parameters to compute distances
    void setParameters( float ax, float bx, float cx ) noexcept {
      m_ax = ax;
      m_bx = bx;
      m_cx = cx;
    }
    float ax() const noexcept { return m_ax; }
    float bx() const noexcept { return m_bx; }
    float cx() const noexcept { return m_cx; }

    //**********Update the parameters of a track iteratively in the Fit
    void updateParameters( float dax, float dbx, float dcx ) {
      m_ax += dax;
      m_bx += dbx;
      m_cx += dcx;
    }

    void  setXT1( float value ) { m_xT1 = value; }
    float xT1() const noexcept { return m_xT1; }
    void  setXT3( float value ) { m_xT3 = value; }
    float xT3() const noexcept { return m_xT3; }

    // validity
    void setValid( bool v ) { m_valid = v; }
    bool valid() const noexcept { return m_valid; }

    void setRecovered( bool v ) { m_recovered = v; }
    bool recovered() const noexcept { return m_recovered; }

    // dRatio
    void  setdRatio( float dRatio ) { m_dRatio = dRatio; }
    float dRatio() const noexcept { return m_dRatio; }

    // chi2
    void  setChi2PerDoF( float chi2, int nDoF ) { m_chi2PerDoF = chi2 / nDoF; }
    void  setChi2PerDoF( float chi2PerDoF ) { m_chi2PerDoF = chi2PerDoF; }
    float chi2PerDoF() const noexcept { return m_chi2PerDoF; }

    // X at a given Z
    float x( float z ) const noexcept {
      const float dz = z - zReference;
      return m_ax + dz * ( m_bx + dz * m_cx * ( 1.f + m_dRatio * dz ) );
    }
    // X at a given Z
    float xFromDz( float dz ) const noexcept { return m_ax + dz * ( m_bx + dz * m_cx * ( 1.f + m_dRatio * dz ) ); }
    // X at a given Z
    float xFromDz( float dz, float dz2Corr ) const noexcept { return m_ax + dz * m_bx + m_cx * dz2Corr; }

    // Slope X-Z plane track at a given z
    float xSlope( float z ) const noexcept {
      float dz = z - Pr::Hybrid::zReference;
      return m_bx + 2.f * dz * m_cx + 3.f * dz * dz * m_cx * m_dRatio; // is it need?
    }
    float xSlopeFromDz( float dz ) const noexcept {
      return m_bx + 2.f * dz * m_cx + 3.f * dz * dz * m_cx * m_dRatio; // is it need?
    }
    // Slope X-Z plane track at z reference
    float xSlope0() const noexcept {
      return m_bx; // is it need?
    }

  protected:
    // Global Protected variables
    float m_chi2PerDoF{ 0.f };
    bool  m_valid{ true };
    bool  m_recovered{ false };
    float m_ax{ 0.f };
    float m_bx{ 0.f };
    float m_cx{ 0.f };
    float m_dRatio{ 0.f };
    float m_xT1{ 0.f };
    float m_xT3{ 0.f };
  };
  //    //Specialised class for X candidates
  class SeedTrackX final : public AbsSeedTrack {
  public:
    SeedTrackX()                                = delete;
    SeedTrackX( const SeedTrackX& )             = default;
    SeedTrackX( SeedTrackX&& )                  = delete;
    SeedTrackX& operator=( const SeedTrackX& )  = default;
    SeedTrackX& operator=( SeedTrackX&& other ) = delete;
    SeedTrackX( float dratio ) : AbsSeedTrack( dratio ) {}
    SeedTrackX( float dratio, const SeedTrackHitsX& hits ) : AbsSeedTrack( dratio ), m_hits( hits ) {}

    auto&       chi2Hits() { return m_chi2Hits; }
    const auto& chi2Hits() const noexcept { return m_chi2Hits; }
    // Handling the hits on the track ( PrHits is vector<PrHit*> )
    SeedTrackHitsX&       hits() noexcept { return m_hits; };
    const SeedTrackHitsX& hits() const noexcept { return m_hits; }
    void                  addHit( const ModPrHit& hit ) noexcept { m_hits.push_back( hit ); };
    // distance from Hit of the track
    float distance( const ModPrHit& hit, const float z ) const noexcept { return hit.coord - x( z ); }
    float distanceFromDz( const ModPrHit& hit, const float dz ) const noexcept { return hit.coord - xFromDz( dz ); }

    float chi2( const FT::Hits& sciFiHits, const ModPrHit& hit ) const noexcept {
      const auto  i = hit.fullDex;
      const float d = distance( hit, sciFiHits.z( i ) );
      float       w = sciFiHits.w( i );
      return d * d * w;
    }
    template <typename Operator>
    struct CompareBySize {
      bool operator()( const SeedTrackX& lhs, const SeedTrackX& rhs ) const noexcept {
        constexpr auto cmp = Operator{};
        return cmp( std::forward_as_tuple( lhs.hits().size(), rhs.chi2PerDoF() ),
                    std::forward_as_tuple( rhs.hits().size(), lhs.chi2PerDoF() ) );
      }
    };
    static constexpr auto LowerBySize   = CompareBySize<std::greater<>>{};
    static constexpr auto GreaterBySize = CompareBySize<std::less<>>{};

    unsigned int size() const noexcept { return m_hits.size(); }

    // Global Private variables
  private:
    SeedTrackHitsX                                       m_hits{};
    std::array<float, LHCb::Detector::FT::nXLayersTotal> m_chi2Hits{};
  };

  using SeedTracksX = std::vector<SeedTrackX>;

  //    //Specialised class for XY candidates
  class SeedTrack final : public AbsSeedTrack {
  public:
    SeedTrack()                               = delete;
    SeedTrack( const SeedTrack& )             = default;
    SeedTrack( SeedTrack&& )                  = delete;
    SeedTrack& operator=( const SeedTrack& )  = default;
    SeedTrack& operator=( SeedTrack&& other ) = delete;
    SeedTrack( float dratio ) : AbsSeedTrack( dratio ) {}
    SeedTrack( const SeedTrackX& xTrack )
        : AbsSeedTrack( xTrack ), m_hits( xTrack.hits().begin(), xTrack.hits().end() ) {}

    auto&       chi2Hits() { return m_chi2Hits; }
    const auto& chi2Hits() const noexcept { return m_chi2Hits; }

    // Handling the hits on the track ( PrHits is vector<PrHit*> )
    SeedTrackHits&       hits() noexcept { return m_hits; };
    const SeedTrackHits& hits() const noexcept { return m_hits; }
    void                 addHit( const ModPrHit& hit ) noexcept { m_hits.push_back( hit ); };
    void                 setParameters( float ax, float bx, float cx, float ay, float by ) noexcept {
      m_ax = ax;
      m_bx = bx;
      m_cx = cx;
      m_ay = ay;
      m_by = by;
    }
    float ay() const noexcept { return m_ay; }
    float by() const noexcept { return m_by; }

    //**********Update the parameters of a track iteratively in the Fit
    void updateParameters( float dax, float dbx, float dcx, float day = 0.f, float dby = 0.f ) {
      m_ax += dax;
      m_bx += dbx;
      m_cx += dcx;
      m_ay += day;
      m_by += dby;
    }
    void setYParam( float ay = 0.f, float by = 0.f ) {
      m_ay = ay;
      m_by = by;
    }
    void setnXnY( unsigned int nx, unsigned int ny ) {
      m_nx = nx;
      m_ny = ny;
    }
    unsigned int nx() const noexcept { return m_nx; }
    unsigned int ny() const noexcept { return m_ny; }

    // BackProjection
    float y( float z ) const noexcept { return ( m_ay + m_by * ( z - zReference ) ); }
    float yFromDz( float dz ) const noexcept { return ( m_ay + m_by * dz ); }
    float yRef() const noexcept { return m_ay; }
    float y0() const noexcept { return m_ay - m_by * zReference; }
    // Slope by
    float ySlope() const noexcept { return m_by; }
    // y positon on Track
    float yOnTrack( const FT::Hits& sciFiHits, unsigned fullDex ) const noexcept {
      return sciFiHits.yOnTrack( fullDex, y0(), m_by );
    }
    // distance from Hit of the track
    float distance( const FT::Hits& sciFiHits, unsigned fullDex ) const noexcept {
      float yTra = yOnTrack( sciFiHits, fullDex ); // is it needed for x - layers?
      return sciFiHits.x( fullDex, yTra ) - x( sciFiHits.z( fullDex, yTra ) );
    }
    float chi2( const FT::Hits& sciFiHits, unsigned fullDex ) const noexcept {
      const float d = distance( sciFiHits, fullDex );
      float       w = sciFiHits.w( fullDex );
      return d * d * w;
    }

    template <typename Operator>
    struct CompareBySize {
      bool operator()( const SeedTrack& lhs, const SeedTrack& rhs ) const noexcept {
        constexpr auto cmp = Operator{};
        return cmp( std::forward_as_tuple( lhs.hits().size(), rhs.chi2PerDoF() ),
                    std::forward_as_tuple( rhs.hits().size(), lhs.chi2PerDoF() ) );
      }
    };
    static constexpr auto LowerBySize   = CompareBySize<std::greater<>>{};
    static constexpr auto GreaterBySize = CompareBySize<std::less<>>{};

    unsigned int size() const noexcept { return m_hits.size(); }

  private:
    unsigned int                                        m_nx{};
    unsigned int                                        m_ny{};
    float                                               m_ay{};
    float                                               m_by{};
    SeedTrackHits                                       m_hits{};
    std::array<float, LHCb::Detector::FT::nLayersTotal> m_chi2Hits{};
  };

  using SeedTracks = std::vector<SeedTrack>;

} // namespace LHCb::Pr::Hybrid
