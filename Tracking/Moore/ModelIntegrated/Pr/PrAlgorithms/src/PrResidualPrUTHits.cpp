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
// Include files
#include "Event/ODIN.h"
#include "Event/PrHits.h"
#include "Event/PrLongTracks.h"
#include "Event/PrUpstreamTracks.h"
#include "UTDAQ/UTInfo.h"
#include "UTDet/DeUTDetector.h"

#include "Gaudi/Accumulators.h"
#include "GaudiKernel/IRegistry.h"
#include "LHCbAlgs/Transformer.h"

#include "boost/dynamic_bitset.hpp"

#include <memory>
#include <vector>

//-----------------------------------------------------------------------------
// class : PrResidualPrUTHits
// Store residual PrUTHits after other Algorithms, e.g. PrMatchNN/PrForward used
//
// 2020-04-21 : Peilian Li
//
//-----------------------------------------------------------------------------

template <typename T>
class PrResidualPrUTHits : public LHCb::Algorithm::Transformer<LHCb::Pr::UT::Hits( const EventContext&, const T&,
                                                                                   const LHCb::Pr::UT::Hits& )> {

public:
  using base_class_t =
      LHCb::Algorithm::Transformer<LHCb::Pr::UT::Hits( const EventContext&, const T&, const LHCb::Pr::UT::Hits& )>;

  LHCb::Pr::UT::Hits operator()( const EventContext&, const T&, const LHCb::Pr::UT::Hits& ) const override;

  PrResidualPrUTHits( const std::string& name, ISvcLocator* pSvcLocator )
      : base_class_t( name, pSvcLocator,
                      { typename base_class_t::KeyValue{ "TracksLocation", "" },
                        typename base_class_t::KeyValue{ "PrUTHitsLocation", "" } },
                      typename base_class_t::KeyValue{ "PrUTHitsOutput", "" } ) {}
};

// Declaration of the Algorithm Factory
DECLARE_COMPONENT_WITH_ID( PrResidualPrUTHits<LHCb::Pr::Long::Tracks>, "PrResidualPrUTHits" )
DECLARE_COMPONENT_WITH_ID( PrResidualPrUTHits<LHCb::Pr::Upstream::Tracks>, "PrResidualPrUTHits_Upstream" )

//=============================================================================
// Main execution
//=============================================================================
template <typename T>
LHCb::Pr::UT::Hits PrResidualPrUTHits<T>::operator()( const EventContext& evtCtx, const T& tracks,
                                                      const LHCb::Pr::UT::Hits& uthithandler ) const {
  LHCb::Pr::UT::Hits tmp{ Zipping::generateZipIdentifier(), LHCb::getMemResource( evtCtx ) };

  // mark used UT hits
  const unsigned int nhits = uthithandler.nHits();
  tmp.reserve( nhits );
  boost::dynamic_bitset<> used{ nhits, false };

  /// mark used SciFi Hits
  for ( const auto& track : tracks.scalar() ) {
    const int nuthits = track.nUTHits().cast();
    for ( int idx = 0; idx < nuthits; idx++ ) {
      const int index = track.ut_index( idx ).cast();
      if ( index >= 0 ) used[index] = true;
    }
  }

  for ( auto fullchan = 0; fullchan < static_cast<int>( UTInfo::MaxNumberOfSectors ); fullchan++ ) {
    const auto indexs = uthithandler.indices( fullchan );

    for ( int idx = indexs.first; idx != indexs.second; idx++ ) {
      if ( used[idx] ) continue;
      tmp.copyHit( fullchan, idx, uthithandler );
    }
  }
  return tmp;
}
