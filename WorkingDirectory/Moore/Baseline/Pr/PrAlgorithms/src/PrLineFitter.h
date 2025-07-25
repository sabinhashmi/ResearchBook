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
#ifndef PRLINEFITTER_H
#define PRLINEFITTER_H 1

// Include files
#include "PrKernel/PrHit.h"
#include "PrKernel/PrSciFiHits.h"

/** @class PrLineFitter PrLineFitter.h
 *  Simple class to fit a line with coordinates
 *
 *  @author Olivier Callot
 *  @date   2012-08-03
 */
template <typename Container>
class PrLineFitter final {
public:
  using PrSciFiHits = SciFiHits::PrSciFiHits;
  void reset( float z, Container* hits ) {
    m_z0    = z;
    m_s0    = 0.f;
    m_sz    = 0.f;
    m_sz2   = 0.f;
    m_sc    = 0.f;
    m_scz   = 0.f;
    m_c0    = 0.f;
    m_tc    = 0.f;
    m_ihits = hits;
  }

  template <typename T>
  void addHit( const T& hit, float coord, const PrSciFiHits& SciFiHits ) {
    addHitInternal( hit, coord, SciFiHits );
  }

  template <typename T>
  float distance( const T& hit, const PrSciFiHits& SciFiHits ) {
    return hit.coord() - ( m_c0 + ( SciFiHits.z( hit.fulldex() ) - m_z0 ) * m_tc );
  }

  template <typename T>
  float chi2( const T& hit, const PrSciFiHits& SciFiHits ) {
    float d = distance( hit, SciFiHits );
    return d * d * SciFiHits.w( hit.fulldex() );
  }

  float coordAtRef() const { return m_c0; }
  float slope() const { return m_tc; }

  void solve() {
    float den = ( m_sz * m_sz - m_s0 * m_sz2 );
    m_c0      = ( m_scz * m_sz - m_sc * m_sz2 ) / den;
    m_tc      = ( m_sc * m_sz - m_s0 * m_scz ) / den;
  }

private:
  template <typename T>
  void addHitInternal( const T& hit, float c, const PrSciFiHits& SciFiHits ) {
    const auto hitindex = [&]() {
      if constexpr ( std::is_integral_v<T> ) {
        return hit;
      } else {
        return hit.fulldex();
      }
    }();
    m_ihits->push_back( hitindex );
    const float w = SciFiHits.w( hitindex );
    const float z = SciFiHits.z( hitindex ) - m_z0;
    m_s0 += w;
    m_sz += w * z;
    m_sz2 += w * z * z;
    m_sc += w * c;
    m_scz += w * c * z;
  }

  float      m_z0    = 0.;
  Container* m_ihits = nullptr;
  float      m_c0    = 0.;
  float      m_tc    = 0.;

  float m_s0  = 0.;
  float m_sz  = 0.;
  float m_sz2 = 0.;
  float m_sc  = 0.;
  float m_scz = 0.;
};
#endif // PRLINEFITTER_H
