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

#include "DetDesc/GenericConditionAccessorHolder.h"
#include "Event/PrHits.h"
#include "Event/RawBank.h"
#include "Event/UTHitCluster.h"
#include "Kernel/IUTReadoutTool.h"
#include "Kernel/UTDecoder.h"
#include "LHCbAlgs/Transformer.h"
#include "PrKernel/UTGeomCache.h"
#include "PrKernel/UTHitHandler.h"
#include "UTDAQ/UTDAQHelper.h"
#include "UTDAQ/UTInfo.h"
#include "UTDet/DeUTDetector.h"
#include <cassert>

namespace LHCb::Pr::UT {

  template <typename HANDLER>
  using Transformer    = LHCb::Algorithm::Transformer<HANDLER( const EventContext&, const RawBank::View&,
                                                               const RawBank::View&, const UTGeomCache& ),
                                                   LHCb::Algorithm::Traits::usesConditions<UTGeomCache>>;
  using PositionMethod = UTDecoder<::UTDAQ::version::v5>::PositionMethod;
  template <typename HANDLER>
  class StoreHit : public Transformer<HANDLER> {
  public:
    using KeyValue = typename Transformer<HANDLER>::KeyValue;
    using Transformer<HANDLER>::inputLocation;

    StoreHit( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer<HANDLER>( name, pSvcLocator,
                                { KeyValue{ "UTRawBank", "DAQ/RawBanks/UT" },
                                  KeyValue{ "UTErrorRawBank", "DAQ/RawBanks/UTError" },
                                  KeyValue{ "GeomCache", "AlgorithmSpecific-" + name + "-UTGeomCache" } },
                                KeyValue{ "UTHitsLocation", UTInfo::HitLocation } ) {}

    StatusCode initialize() override {
      return Transformer<HANDLER>::initialize().andThen( [&] {
        // TODO : alignment need the updateSvc for detector ( UT experts needed )
        this->addConditionDerivation( { DeUTDetLocation::location(), m_readoutTool->getReadoutInfoKey() },
                                      this->template inputLocation<UTGeomCache>(),
                                      [this]( const DeUTDetector& utDet, IUTReadoutTool::ReadoutInfo const& roInfo ) {
                                        return UTGeomCache{ utDet, *m_readoutTool, roInfo };
                                      } );
      } );
    }

    HANDLER operator()( const EventContext& evtCtx, const RawBank::View& utBanks, const RawBank::View& utErrorBanks,
                        const UTGeomCache& cache ) const override {
      HANDLER hitHandler{ Zipping::generateZipIdentifier(), LHCb::getMemResource( evtCtx ) };
      hitHandler.reserve( 10000 );

      // only process requested bank types via m_bankTypes
      if ( m_useUTBanks ) {
        try {
          for ( const auto& bank : utBanks ) {

            // make local decoder
            if ( bank->size() == 0 ) continue;
            // Check if the source id is MC or data
            auto decode = [&hitHandler,
                           geomOffset = UTDAQ::boardIDfromSourceID( bank->sourceID() ) *
                                        static_cast<unsigned>( UTInfo::SectorNumbers::MaxSectorsPerBoard ),
                           &cache]( auto decoder_range, auto clusterChargeHandler, auto fracStripBitsHandler ) {
              for ( const auto& aWord : decoder_range ) {
                const std::size_t geomIdx = geomOffset + ( aWord.channelID() / 512 );
                assert( geomIdx < cache.sectors.size() );
                assert( geomIdx < cache.fullchan.size() );

                auto        aSector  = cache.sectors[geomIdx];
                const auto& fullChan = cache.fullchan[geomIdx];

                const auto strip = ( aWord.channelID() & 0x1ff );

#ifdef USE_DD4HEP
                hitHandler.emplace_back( aSector, fullChan.idx, strip, fracStripBitsHandler( aWord ),
#else
                hitHandler.emplace_back( *aSector, fullChan.idx, strip, fracStripBitsHandler( aWord ),
#endif
                                         Detector::UT::ChannelID{ fullChan.chanID + strip }, aWord.pseudoSizeBits(),
                                         aWord.hasHighThreshold(), clusterChargeHandler( aWord ) );
              }
            };
            switch ( ::UTDAQ::version{ bank->version() } ) {
            case ::UTDAQ::version::v2:
              if ( !m_isCluster )
                decode(
                    UTDecoder<::UTDAQ::version::v2>{ *bank }.posRange(),
                    []( const auto& aWord ) { return aWord.clusterCharge(); },
                    []( const auto& aWord ) { return aWord.fracStripBits(); } );
              else
                decode(
                    UTDecoder<::UTDAQ::version::v2>{ *bank }.posAdcRange( m_positionMethod, m_stripMax ),
                    []( const auto& aWord ) { return aWord.clusterCharge(); },
                    []( const auto& aWord ) { return aWord.fracStripBits(); } );
              break;
            case ::UTDAQ::version::v5:
              if ( !m_isCluster )
                decode(
                    UTDecoder<::UTDAQ::version::v5>{ *bank }.posRange(),
                    []( const auto& aWord ) { return aWord.clusterCharge(); },
                    []( const auto& aWord ) { return aWord.fracStripBits(); } );
              else
                decode(
                    UTDecoder<::UTDAQ::version::v5>{ *bank }.posAdcRange( m_positionMethod, m_stripMax ),
                    []( const auto& aWord ) { return aWord.clusterCharge(); },
                    []( const auto& aWord ) { return aWord.fracStripBits(); } );
              break;
            case ::UTDAQ::version::v4:
              decode(
                  UTDecoder<::UTDAQ::version::v4>{ *bank }.posRange(), [&]( const auto& ) { return 0; },
                  []( const auto& aWord ) { return aWord.fracStripBits() / 4; } );
              break;
            default:
              throw std::runtime_error{ "unknown version of the RawBank" }; /* OOPS: unknown format */
            };
          }
          m_nUTBanks += utBanks.size();
        } catch ( std::runtime_error& ) { // FIXME: temporary work around for MC produced in dec 2020 - may 2021 -- for
                                          // FEST only!
          hitHandler.clear();
          ++m_bad_data;
        }
      }
      if ( m_useUTErrorBanks ) {
        try {
          for ( const auto& bank : utErrorBanks ) {

            // make local decoder
            if ( bank->size() == 0 ) continue;
            // Check if the source id is MC or data
            auto decode = [&hitHandler,
                           geomOffset = UTDAQ::boardIDfromSourceID( bank->sourceID() ) *
                                        static_cast<unsigned>( UTInfo::SectorNumbers::MaxSectorsPerBoard ),
                           &cache]( auto decoder_range, auto clusterChargeHandler, auto fracStripBitsHandler ) {
              for ( const auto& aWord : decoder_range ) {
                const std::size_t geomIdx = geomOffset + ( aWord.channelID() / 512 );
                assert( geomIdx < cache.sectors.size() );
                assert( geomIdx < cache.fullchan.size() );

                auto        aSector  = cache.sectors[geomIdx];
                const auto& fullChan = cache.fullchan[geomIdx];

                const auto strip     = ( aWord.channelID() & 0x1ff );
                auto       channelId = Detector::UT::ChannelID{ fullChan.chanID + strip };

#ifdef USE_DD4HEP
                hitHandler.emplace_back( aSector, fullChan.idx, strip, fracStripBitsHandler( aWord ),
#else
                hitHandler.emplace_back( *aSector, fullChan.idx, strip, fracStripBitsHandler( aWord ),
#endif
                                         channelId, aWord.pseudoSizeBits(), aWord.hasHighThreshold(),
                                         clusterChargeHandler( aWord ) );
              }
            };
            switch ( ::UTDAQ::version{ bank->version() } ) {
            case ::UTDAQ::version::v2:
              if ( !m_isCluster )
                decode(
                    UTDecoder<::UTDAQ::version::v2>{ *bank }.posRange(),
                    []( const auto& aWord ) { return aWord.clusterCharge(); },
                    []( const auto& aWord ) { return aWord.fracStripBits(); } );
              else
                decode(
                    UTDecoder<::UTDAQ::version::v2>{ *bank }.posAdcRange( m_positionMethod, m_stripMax ),
                    []( const auto& aWord ) { return aWord.clusterCharge(); },
                    []( const auto& aWord ) { return aWord.fracStripBits(); } );
              break;
            case ::UTDAQ::version::v5:
              if ( !m_isCluster )
                decode(
                    UTDecoder<::UTDAQ::version::v5>{ *bank }.posRange(),
                    []( const auto& aWord ) { return aWord.clusterCharge(); },
                    []( const auto& aWord ) { return aWord.fracStripBits(); } );
              else
                decode(
                    UTDecoder<::UTDAQ::version::v5>{ *bank }.posAdcRange( m_positionMethod, m_stripMax ),
                    []( const auto& aWord ) { return aWord.clusterCharge(); },
                    []( const auto& aWord ) { return aWord.fracStripBits(); } );
              break;
            case ::UTDAQ::version::v4:
              decode(
                  UTDecoder<::UTDAQ::version::v4>{ *bank }.posRange(), [&]( const auto& ) { return 0; },
                  []( const auto& aWord ) { return aWord.fracStripBits() / 4; } );
              break;
            default:
              throw std::runtime_error{ "unknown version of the RawBank" }; /* OOPS: unknown format */
            };
          }
          m_nUTErrorBanks += utErrorBanks.size();

        } catch ( std::runtime_error& ) { // FIXME: temporary work around for MC produced in dec 2020 - may 2021 -- for
                                          // FEST only!
          hitHandler.clear();
          ++m_bad_data;
        }
      }

      if constexpr ( std::is_same_v<HANDLER, ::LHCb::Pr::UT::Hits> ) hitHandler.addPadding();
      if constexpr ( std::is_same_v<HANDLER, UTHitClusters> ) hitHandler.forceSort();

      if ( !m_assumeSorted.value() ) {
        if constexpr ( !std::is_same_v<HANDLER, UTHitClusters> ) {
          throw GaudiException( "Sorting of the SoA hit container (or HitHandler) is not implemented, yet.",
                                "PrStoreUTHit", StatusCode::FAILURE );
        } // UTHitClusters are *always* sorted explicitly
      }

      if ( !m_assumeUnique.value() ) {
        if constexpr ( std::is_same_v<HANDLER, UTHitClusters> ) {
          auto is_unique = hitHandler.forceUnique();
          m_not_unique_clusters += ( !is_unique );
        } else
          throw GaudiException( "Forcing a unique SoA hit container or HitHandler is not implemented, yet.",
                                "PrStoreUTHit", StatusCode::FAILURE );
      }

      return hitHandler;
    }

  private:
    //---Properties
    Gaudi::Property<bool> m_useUTBanks{ this, "UseUTBanks", true, "Whether to decode RawBank::UT" };
    Gaudi::Property<bool> m_useUTErrorBanks{ this, "UseUTErrorBanks", true, "Whether to decode RawBank::UTError" };
    Gaudi::Property<bool> m_isCluster{ this, "isCluster", true };
    Gaudi::Property<bool> m_assumeSorted{ this, "AssumeSorted", true };
    Gaudi::Property<bool> m_assumeUnique{ this, "AssumeUnique", true };
    Gaudi::Property<PositionMethod> m_positionMethod{ this, "positionMethod", PositionMethod::AdcWeighting };
    Gaudi::Property<unsigned int>   m_stripMax{ this, "stripMax", 128 };
    mutable Gaudi::Accumulators::SummingCounter<>  m_nUTBanks{ this, "# RawBank::UT banks" };
    mutable Gaudi::Accumulators::SummingCounter<>  m_nUTErrorBanks{ this, "# RawBank::UTError banks" };
    mutable Gaudi::Accumulators::BinomialCounter<> m_not_unique_clusters{ this, "Non-unique UT clusters in event" };

    mutable Gaudi::Accumulators::MsgCounter<MSG::ERROR> m_bad_data{ this, "Decoding Error -- dropping all UT hits" };
    ToolHandle<IUTReadoutTool>                          m_readoutTool{ this, "ReadoutTool", "UTReadoutTool" };
    bool                                                m_add_ut_banks;
    bool                                                m_add_ut_error_banks;
  };
  // Declaration of the Algorithm Factory
  DECLARE_COMPONENT_WITH_ID( StoreHit<::UT::HitHandler>, "PrStoreUTHit" )      // scalar hits
  DECLARE_COMPONENT_WITH_ID( StoreHit<Hits>, "PrStorePrUTHits" )               // SoA hits
  DECLARE_COMPONENT_WITH_ID( StoreHit<UTHitClusters>, "PrStoreUTHitClusters" ) // scalar hits

} // namespace LHCb::Pr::UT
