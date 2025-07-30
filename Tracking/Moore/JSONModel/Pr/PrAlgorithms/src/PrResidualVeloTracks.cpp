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
#include "Gaudi/Accumulators.h"
#include "GaudiKernel/IRegistry.h"
#include "LHCbAlgs/Transformer.h"
#include <algorithm>
#include <array>
#include <vector>

#include "Detector/VP/VPChannelID.h"
#include "Kernel/STLExtensions.h"
#include "PrKernel/VeloPixelInfo.h"
#include "VPDet/DeVP.h"

#include "Event/PrLongTracks.h"
#include "Event/PrVeloHits.h"
#include "Event/PrVeloTracks.h"

#include "Event/ODIN.h"
#include "LHCbMath/SIMDWrapper.h"
#include <Vc/Vc>

#include "Kernel/AllocatorUtils.h"
#include "boost/container/small_vector.hpp"
#include "boost/container/static_vector.hpp"
#include "boost/dynamic_bitset.hpp"
#include <memory>

//-----------------------------------------------------------------------------
// class : PrResidualVeloTracks
// Store residual VeloTracks after other Algorithms, e.g. PrMatchNN used
//
// 2020-04-02 : Peilian Li
//
//-----------------------------------------------------------------------------

using LongTracks = LHCb::Pr::Long::Tracks;
using VeloTracks = LHCb::Pr::Velo::Tracks;
class PrResidualVeloTracks
    : public LHCb::Algorithm::Transformer<LHCb::Pr::Velo::Tracks( const LongTracks&, const VeloTracks& )> {

public:
  PrResidualVeloTracks( const std::string& name, ISvcLocator* pSvcLocator );

  LHCb::Pr::Velo::Tracks operator()( const LongTracks&, const VeloTracks& ) const override;
};

// Declaration of the Algorithm Factory
DECLARE_COMPONENT_WITH_ID( PrResidualVeloTracks, "PrResidualVeloTracks" )

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
PrResidualVeloTracks::PrResidualVeloTracks( const std::string& name, ISvcLocator* pSvcLocator )
    : Transformer( name, pSvcLocator,
                   { KeyValue{ "TracksLocation", "" }, KeyValue{ "VeloTrackLocation", "Rec/Track/Velo" } },
                   KeyValue{ "VeloTrackOutput", "" } ) {}

//=============================================================================
// Main execution
//=============================================================================
LHCb::Pr::Velo::Tracks PrResidualVeloTracks::operator()( const LongTracks& tracks,
                                                         const VeloTracks& velotracks ) const {

  using simd = SIMDWrapper::scalar::types;

  LHCb::Pr::Velo::Tracks tmp;
  tmp.reserve( velotracks.size() );

  if ( velotracks.empty() ) {
    if ( msgLevel( MSG::DEBUG ) )
      debug() << "Velo Track container '" << inputLocation<VeloTracks>() << "' is empty" << endmsg;
    return tmp;
  }

  const unsigned int      nvelo = velotracks.size();
  boost::dynamic_bitset<> used{ nvelo, false };

  auto const veloiter = velotracks.scalar();
  auto const longiter = tracks.scalar();
  for ( auto const& track : longiter ) {
    const auto veloidx = track.trackVP().cast();
    used[veloidx]      = true;
  }
  for ( auto const& velotrack : veloiter ) {
    const int t = velotrack.offset();
    if ( used[t] ) continue;
    auto mask = ( !used[t] );
    tmp.copy_back<simd>( velotracks, t, mask );
  }
  return tmp;
}
