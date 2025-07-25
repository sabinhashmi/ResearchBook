/*****************************************************************************\
* (c) Copyright 2000-2022 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// Gaudi
#include "LHCbAlgs/Transformer.h"

// LHCb
#include "Event/MuonPID.h"
#include "Event/MuonPIDs_v2.h"
#include "Event/RelationTable.h"
#include "Event/Track.h"

/**
 * Converter from MuonPID SoA PoD to LHCb::MuonPIDs (KeyedContainer)
 *
 * @author Ricardo Vazquez Gomez (UB)
 *
 * Based on https://gitlab.cern.ch/lhcb/Rec/blob/master/Pr/PrConverters/src/fromPrVeloUTTrack.cpp
 * from Michel De Cian
 */

namespace LHCb::Converters::Muon {

  template <typename Relations>
  class fromV2MuonPIDV1MuonPID : public Algorithm::Transformer<LHCb::MuonPIDs(
                                     const Event::v2::Muon::PIDs&, const Relations&, const LHCb::Tracks& )> {
  public:
    using KeyValue = typename fromV2MuonPIDV1MuonPID<Relations>::KeyValue;
    fromV2MuonPIDV1MuonPID( const std::string& name, ISvcLocator* pSvcLocator )
        : fromV2MuonPIDV1MuonPID<Relations>::Transformer( name, pSvcLocator,
                                                          { KeyValue( "InputMuonPIDs", "" ),
                                                            KeyValue( "InputTrackRelations", "" ),
                                                            KeyValue( "InputMuonTracks", "" ) },
                                                          { KeyValue( "OutputMuonPIDs", "" ) } ) {}

    LHCb::MuonPIDs operator()( const Event::v2::Muon::PIDs& muonPIDs, const Relations& trackrels,
                               const LHCb::Tracks& muontracks ) const override {
      LHCb::MuonPIDs out;
      out.reserve( muonPIDs.size() );

      m_nbMuonPIDsCounter += muonPIDs.size();

      for ( auto const& muonPID : muonPIDs.scalar() ) {
        auto newMuonPID = new LHCb::MuonPID();
        newMuonPID->setChi2Corr( muonPID.Chi2Corr().cast() );
        newMuonPID->setIsMuon( muonPID.IsMuon().cast() );
        newMuonPID->setIsMuonTight( muonPID.IsMuonTight().cast() );
        newMuonPID->setInAcceptance( muonPID.InAcceptance().cast() );
        newMuonPID->setPreSelMomentum( muonPID.PreSelMomentum().cast() );
        newMuonPID->setMuonLLMu( muonPID.LLMu().cast() );
        newMuonPID->setMuonLLBg( muonPID.LLBg().cast() );
        newMuonPID->setMuonMVA2( muonPID.CatBoost().cast() );
        // get the correct track
        auto idx = muonPID.indices().cast();
        newMuonPID->setIDTrack( trackrels.get( idx ) );
        if ( muonPID.nTileIDs() > 0 ) newMuonPID->setMuonTrack( muontracks( idx ) );
        out.insert( newMuonPID );
      }
      return out;
    };

  private:
    mutable Gaudi::Accumulators::SummingCounter<> m_nbMuonPIDsCounter{ this, "Nb of Produced MuonPIDs" };
  };

  using RelationsKeyed  = LHCb::Event::V3ToV1Mapping<LHCb::Event::v1::Tracks>;
  using RelationsShared = LHCb::Event::V3ToV1Mapping<LHCb::Event::v1::Track::Selection>;

  DECLARE_COMPONENT_WITH_ID( fromV2MuonPIDV1MuonPID<RelationsKeyed>, "fromV2MuonPIDV1MuonPID" )
  DECLARE_COMPONENT_WITH_ID( fromV2MuonPIDV1MuonPID<RelationsShared>, "fromV2MuonPIDV1MuonPIDShared" )

} // namespace LHCb::Converters::Muon
