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
#include "Event/PrLongTracks.h"
#include "GaudiAlg/FunctionalUtilities.h"
#include "PrKernel/IPrAddUTHitsTool.h" // Interface

#include "GaudiAlg/FunctionalTool.h"
#include "GaudiAlg/GaudiTool.h"
#include "GaudiKernel/IBinder.h"
#include "boost/container/static_vector.hpp"

namespace LHCb::Pr {

  class PrIgnoreUTHitsTool : public Gaudi::Functional::ToolBinder<Gaudi::Interface::Bind::Box<IPrAddUTHitsTool>()> {
  private:
    struct BoundInstance final : Gaudi::Interface::Bind::Stub<IPrAddUTHitsTool> {
      BoundInstance( const PrIgnoreUTHitsTool* ) {}
      void addUTHits( Long::Tracks& ) const override {}
      void getUTHits( const LHCb::State&,
                      boost::container::static_vector<LHCb::LHCbID, TracksInfo::MaxUTHits>& ) const override {}
    };

  public:
    PrIgnoreUTHitsTool( std::string type, std::string name, const IInterface* parent )
        : ToolBinder{ std::move( type ), name, parent, {}, construct<BoundInstance>( this ) } {}
  };
  // Declaration of the Algorithm Factory
  DECLARE_COMPONENT_WITH_ID( PrIgnoreUTHitsTool, "PrIgnoreUTHitsTool" )

} // namespace LHCb::Pr
