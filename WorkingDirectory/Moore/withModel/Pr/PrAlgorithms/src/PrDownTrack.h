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

#include "GaudiKernel/Point3DTypes.h"

#include "Event/PrHits.h"
#include "Event/State.h"
#include "Event/StateParameters.h"

#include "boost/container/small_vector.hpp"

#include <cassert>

namespace Downstream {
  struct Hit {
  private:
    inline auto getScalarHit() const {
      const auto myHits = hits->scalar();
      assert( (std::size_t)hit < myHits.size() );
      return myHits[hit];
    }

  public:
    const LHCb::Pr::UT::Hits* hits = nullptr;
    int                       hit{ 0 };
    float                     x{ 0 }, z{ 0 };
    float                     projection{ 0 };

    using F = SIMDWrapper::scalar::types::float_v;
    using I = SIMDWrapper::scalar::types::int_v;

    Hit( const LHCb::Pr::UT::Hits* _hits, const int _hit, float _x, float _z, float _proj )
        : hits( _hits ), hit( _hit ), x( _x ), z( _z ), projection( _proj ) {}

    [[nodiscard]] auto lhcbID() const {
      const auto mH     = getScalarHit();
      const auto chanID = mH.get<LHCb::Pr::UT::UTHitsTag::channelID>().cast();
      return bit_cast<int>( LHCb::LHCbID( LHCb::Detector::UT::ChannelID( chanID ) ).lhcbID() );
    }
    [[nodiscard]] int planeCode() const {
      const auto mH     = getScalarHit();
      auto       lhcbid = mH.get<LHCb::Pr::UT::UTHitsTag::channelID>().cast();
      return ( lhcbid & static_cast<unsigned int>( UTInfo::MasksBits::HalfLayerMask ) ) >>
             static_cast<int>( UTInfo::MasksBits::HalfLayerBits );
    }
    [[nodiscard]] auto weight() const {
      const auto mH = getScalarHit();
      return mH.get<LHCb::Pr::UT::UTHitsTag::weight>().cast();
    }

    [[nodiscard]] auto sin() const {
      const auto mH = getScalarHit();
      return -mH.get<LHCb::Pr::UT::UTHitsTag::dxDy>().cast() * mH.get<LHCb::Pr::UT::UTHitsTag::cos>().cast();
    }

    [[nodiscard]] auto zAtYEq0() const {
      const auto mH = getScalarHit();
      return mH.get<LHCb::Pr::UT::UTHitsTag::zAtYEq0>().cast();
    }
    [[nodiscard]] bool isYCompatible( const float y, const float tol ) const {
      const auto mH = getScalarHit();
      const auto yMin =
          std::min( mH.get<LHCb::Pr::UT::UTHitsTag::yBegin>().cast(), mH.get<LHCb::Pr::UT::UTHitsTag::yEnd>().cast() );
      const auto yMax =
          std::max( mH.get<LHCb::Pr::UT::UTHitsTag::yBegin>().cast(), mH.get<LHCb::Pr::UT::UTHitsTag::yEnd>().cast() );
      return ( ( ( yMin - tol ) <= y ) && ( y <= ( yMax + tol ) ) );
    }
    [[nodiscard]] inline auto xAt( const float y ) const {
      const auto mH = getScalarHit();
      return mH.get<LHCb::Pr::UT::UTHitsTag::xAtYEq0>().cast() + y * mH.get<LHCb::Pr::UT::UTHitsTag::dxDy>().cast();
    }
  };

  using Hits = std::vector<Hit, LHCb::Allocators::EventLocal<Hit>>;

  inline constexpr auto IncreaseByProj = []( const Hit& lhs, const Hit& rhs ) {
    return ( lhs.projection < rhs.projection ? true : //
                 rhs.projection < lhs.projection ? false
                                                 : //
                 lhs.lhcbID() < rhs.lhcbID() );
  };

  inline constexpr auto IncreaseByAbsProj = []( const Hit& lhs, const Hit& rhs ) {
    return ( std::abs( lhs.projection ) < std::abs( rhs.projection ) ? true : //
                 std::abs( rhs.projection ) < std::abs( lhs.projection ) ? false
                                                                         : //
                 lhs.lhcbID() < rhs.lhcbID() );
  };

} // namespace Downstream

/** @class PrDownTrack PrDownTrack.h
 *  Track helper for Downstream track search
 *  Adapted from Pat/PatKShort package
 *  Further adapted for use with PrLongLivedTracking
 *
 *  @author Olivier Callot
 *  @date   2007-10-18
 *
 *  @author Adam Davis
 *  @date   2016-04-10
 *
 *  @author Christoph Hasse (new framework)
 *  @date   2017-03-01
 */

class PrDownTrack final {
public:
  using Hits = boost::container::small_vector<Downstream::Hit, 8, LHCb::Allocators::EventLocal<Downstream::Hit>>;
  // using Hits = boost::container::static_vector<Downstream::Hit, 20>;
  // Until we can put a bound on the number of hits, use a small_vector

  PrDownTrack( Gaudi::TrackVectorF stateVector, float stateZ, float zUT, LHCb::span<const float, 7> magnetParams,
               LHCb::span<const float, 2> yParams, LHCb::span<const float, 3> momPar, float magnetScale )
      : m_stateVector( stateVector ), m_stateZ( stateZ ), m_zUT( zUT ) {

    const auto tx2  = stateTx() * stateTx();
    const auto ty2  = stateTy() * stateTy();
    m_momentumParam = ( momPar[0] + momPar[1] * tx2 + momPar[2] * ty2 ) * magnetScale;

    // -- See PrFitKsParams to see how these coefficients are derived.
    const float zMagnet = magnetParams[0] + magnetParams[1] * ty2 + magnetParams[2] * tx2 +
                          magnetParams[3] * std::abs( stateQoP() ) + /// this is where the old one stopped.
                          magnetParams[4] * std::abs( stateX() ) + magnetParams[5] * std::abs( stateY() ) +
                          magnetParams[6] * std::abs( stateTy() );

    const float dz      = zMagnet - stateZ;
    const float xMagnet = stateX() + dz * stateTx();
    m_slopeX            = xMagnet / zMagnet;
    const float dSlope  = std::abs( m_slopeX - stateTx() );
    const float dSlope2 = dSlope * dSlope;

    const float by = stateY() / ( stateZ + ( yParams[0] * fabs( stateTy() ) * zMagnet + yParams[1] ) * dSlope2 );
    m_slopeY       = by * ( 1. + yParams[0] * fabs( by ) * dSlope2 );

    const float yMagnet = stateY() + dz * by - yParams[1] * by * dSlope2;

    // -- These resolutions are semi-empirical and are obtained by fitting residuals
    // -- with MCHits and reconstructed tracks
    // -- See Tracking &Alignment meeting, 19.2.2015, for the idea
    float errXMag = dSlope2 * 15.0 + dSlope * 15.0 + 3.0;
    float errYMag = dSlope2 * 80.0 + dSlope * 10.0 + 4.0;

    // -- Assume better resolution for SciFi than for OT
    // -- obviously this should be properly tuned...
    errXMag /= 2.0;
    errYMag /= 1.5;

    // errXMag = 0.5  + 5.3*dSlope + 6.7*dSlope2;
    // errYMag = 0.37 + 0.7*dSlope - 4.0*dSlope2 + 11*dSlope2*dSlope;

    m_weightXMag = 1.0 / ( errXMag * errXMag );
    m_weightYMag = 1.0 / ( errYMag * errYMag );

    m_magnet = Gaudi::XYZPoint( xMagnet, yMagnet, zMagnet );

    //=== Save for reference
    m_displX = 0.;
    m_displY = 0.;

    //=== Initialize all other data members
    m_chi2 = 0.;
  }

  /// getters
  float       stateX() const noexcept { return m_stateVector[0]; }
  float       stateY() const noexcept { return m_stateVector[1]; }
  float       stateZ() const noexcept { return m_stateZ; }
  float       stateTx() const noexcept { return m_stateVector[2]; }
  float       stateTy() const noexcept { return m_stateVector[3]; }
  float       stateQoP() const noexcept { return m_stateVector[4]; }
  Hits&       hits() noexcept { return m_hits; }
  const Hits& hits() const noexcept { return m_hits; }
  float       xMagnet() const noexcept { return m_magnet.x(); }
  float       yMagnet() const noexcept { return m_magnet.y(); }
  float       zMagnet() const noexcept { return m_magnet.z(); }
  float       slopeX() const noexcept { return m_slopeX; }
  float       slopeY() const noexcept { return m_slopeY; }
  float       weightXMag() const noexcept { return m_weightXMag; }
  float       weightYMag() const noexcept { return m_weightYMag; }
  float       chi2() const noexcept { return m_chi2; }
  float       zUT() const noexcept { return m_zUT; }

  /// setters
  void setSlopeX( float slopeX ) noexcept { m_slopeX = slopeX; }
  void setChi2( float chi2 ) noexcept { m_chi2 = chi2; }

  // functions
  [[gnu::always_inline]] inline float xAtZ( float z ) const noexcept {
    const float curvature = 1.6e-5 * ( stateTx() - m_slopeX );
    return xMagnet() + ( z - zMagnet() ) * m_slopeX + curvature * ( z - m_zUT ) * ( z - m_zUT );
  }

  // -- this ignores the change in curvature, to be used for small distances
  [[gnu::always_inline]] inline float xAtZLinear( float z, float xStart, float zStart ) const noexcept {
    return xStart + ( z - zStart ) * m_slopeX;
  }

  [[gnu::always_inline]] inline float yAtZ( float z ) const noexcept {
    return yMagnet() + m_displY + ( z - zMagnet() ) * slopeY();
  }

  void updateX( float dx, float dsl ) noexcept {
    m_displX += dx;
    m_magnet = Gaudi::XYZPoint( m_magnet.x() + dx, m_magnet.y(), m_magnet.z() );
    m_slopeX += dsl;
  }
  void updateY( float dy ) noexcept { m_displY += dy; }

  float dxMagnet() const noexcept { return -m_displX; }

  float initialChi2() const noexcept { return m_displX * m_displX * m_weightXMag + m_displY * m_displY * m_weightYMag; }

  float momentum() const noexcept { return m_momentumParam / ( stateTx() - m_slopeX ); }

  float pt() const noexcept {
    const float tx2      = slopeX() * slopeX();
    const float ty2      = slopeY() * slopeY();
    const float sinTrack = sqrt( 1. - 1. / ( 1. + tx2 + ty2 ) );
    return sinTrack * std::abs( momentum() );
  }

  [[gnu::always_inline]] inline float distance( const Downstream::Hit& hit ) const noexcept {
    return hit.xAt( yAtZ( hit.z ) ) - xAtZ( hit.z );
  }

  [[gnu::always_inline]] inline float distanceLinear( const Downstream::Hit& hit, float zStart,
                                                      float xStart ) const noexcept {
    return hit.xAt( yAtZ( hit.z ) ) - xAtZLinear( hit.z, zStart, xStart );
  }

  bool isYCompatible( const float tol ) const noexcept {
    return std ::all_of( m_hits.begin(), m_hits.end(),
                         [&]( const auto& hit ) { return hit.isYCompatible( yAtZ( hit.z ), tol ); } );
  }

  void sortFinalHits() noexcept {
    std::sort( m_hits.begin(), m_hits.end(), []( const Downstream::Hit& lhs, const Downstream::Hit& rhs ) {
      return std::make_tuple( lhs.z, lhs.lhcbID() ) < std::make_tuple( rhs.z, rhs.lhcbID() );
    } );
  }

private:
  // -- using the float versions everywhere
  Gaudi::TrackVectorF m_stateVector;
  float               m_stateZ{ 0 };
  Gaudi::XYZPointF    m_magnet;

  float m_momentumParam{ 0 };
  float m_zUT{ 0 };
  float m_slopeX{ 0 };
  float m_slopeY{ 0 };
  float m_displX{ 0 };
  float m_displY{ 0 };
  float m_weightXMag{ 0 };
  float m_weightYMag{ 0 };
  float m_chi2{ 0 };

  Hits m_hits; /// working list of hits on this track
};

// collection of downstream tracks... From PatDownTrack
using PrDownTracks = std::vector<PrDownTrack>;
