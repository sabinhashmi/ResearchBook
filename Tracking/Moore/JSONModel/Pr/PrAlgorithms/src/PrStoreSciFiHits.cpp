
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
#include "Detector/FT/FTChannelID.h"
#include "Event/FTLiteCluster.h"
#include "Event/PrHits.h"
#include "FTDAQ/FTInfo.h"
#include "FTDet/DeFTDetector.h"
#include "Kernel/LHCbID.h"
#include "LHCbAlgs/Transformer.h"
#include "LHCbMath/bit_cast.h"
#include "PrKernel/FTMatsCache.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <memory>
#include <string>

#include "GaudiKernel/StdArrayAsProperty.h"

/** @class StoreHits PrStoreSciFiHits.cpp
 *
 *  @brief Transforms FTLiteClusters into the input format needed by the PrForwardTracking
 */

namespace LHCb::Pr::FT {
  using FTLiteClusters = FTLiteCluster::FTLiteClusters;
  using MatsCache      = Detector::FT::Cache::MatsCache;

  // TODO: get this from a tool?
  constexpr auto invClusRes2 = [] {
    auto tmp = std::array{ 0.05f, 0.08f, 0.11f, 0.14f, 0.17f, 0.20f, 0.23f, 0.26f, 0.29f };
    for ( std::size_t i{ 0 }; i < tmp.size(); ++i ) {
      const auto inv = 1.f / tmp[i];
      tmp[i]         = inv * inv;
    }
    return tmp;
  }();

  class StoreHits : public Algorithm::Transformer<Hits( const EventContext&, const FTLiteClusters&, const MatsCache& ),
                                                  Algorithm::Traits::usesConditions<MatsCache>> {
  public:
    StoreHits( const std::string& name, ISvcLocator* pSvcLocator )
        : Transformer( name, pSvcLocator,
                       { KeyValue{ "HitsLocation", FTLiteClusterLocation::Default },
                         KeyValue{ "FTMatsCache", std::string{ MatsCache::Location } + name } },
                       KeyValue{ "Output", PrFTInfo::SciFiHitsLocation } ) {}

    StatusCode initialize() override {
      return Transformer::initialize().andThen( [&] {
        addConditionDerivation(
            { DeFTDetectorLocation::Default }, inputLocation<MatsCache>(),
            [parent = this, applyMatContractionCalibration = m_applyMatContractionCalibration]( const DeFT& deft ) {
              return MatsCache{ deft, parent, applyMatContractionCalibration };
            } );
      } );
    }

    Hits operator()( const EventContext&, const FTLiteClusters&, const MatsCache& ) const override;

  private:
    // Calib
    Gaudi::Property<bool> m_applyMatContractionCalibration{ this, "ApplyMatContractionCalibration",
                                                            true }; // TODO: remove this flag when fallback solution for
                                                                    // missing conditions is in place
    Gaudi::Property<std::array<bool, FTConstants::nLayersTotal>> m_layerMasks{ this, "LayerMasks", {} };
    // Counters
    using SC = Gaudi::Accumulators::StatCounter<>;
    using SCbuf =
        Gaudi::Accumulators::Buffer<Gaudi::Accumulators::StatAccumulator, Gaudi::Accumulators::atomicity::full, double>;
    mutable SC                                               m_cntTotalHits{ this, "Total number of hits" };
    mutable std::array<SC, LHCb::Detector::FT::nLayersTotal> m_cntHitsPerLayer{
        SC{ this, "Hits in T1X1" }, SC{ this, "Hits in T1U" }, SC{ this, "Hits in T1V" }, SC{ this, "Hits in T1X2" },
        SC{ this, "Hits in T2X1" }, SC{ this, "Hits in T2U" }, SC{ this, "Hits in T2V" }, SC{ this, "Hits in T2X2" },
        SC{ this, "Hits in T3X1" }, SC{ this, "Hits in T3U" }, SC{ this, "Hits in T3V" }, SC{ this, "Hits in T3X2" } };
    mutable std::array<SC, LHCb::Detector::FT::nLayersTotal> m_cntXPerLayer{
        SC{ this, "Average X in T1X1" }, SC{ this, "Average X in T1U" },  SC{ this, "Average X in T1V" },
        SC{ this, "Average X in T1X2" }, SC{ this, "Average X in T2X1" }, SC{ this, "Average X in T2U" },
        SC{ this, "Average X in T2V" },  SC{ this, "Average X in T2X2" }, SC{ this, "Average X in T3X1" },
        SC{ this, "Average X in T3U" },  SC{ this, "Average X in T3V" },  SC{ this, "Average X in T3X2" } };
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_noCalib{
        this, "Requested to correct FT mats for temperature distortions but conditions not available. "
              "These corrections cannot be applied. Check your conditions tags if you intend to apply "
              "these corrections or set property ApplyMatContractionCalibration to false if you do not. "
              "Continuing without applying corrections." };
  };

  DECLARE_COMPONENT_WITH_ID( StoreHits, "PrStoreSciFiHits" )

  Hits StoreHits::operator()( const EventContext& evtCtx, FTLiteClusters const& clusters,
                              MatsCache const& cache ) const {

    Hits hits{ LHCb::getMemResource( evtCtx ) };

    auto                                                bufTotalHits    = m_cntTotalHits.buffer();
    std::array<SCbuf, LHCb::Detector::FT::nLayersTotal> bufHitsPerLayer = {
        m_cntHitsPerLayer[0].buffer(), m_cntHitsPerLayer[1].buffer(),  m_cntHitsPerLayer[2].buffer(),
        m_cntHitsPerLayer[3].buffer(), m_cntHitsPerLayer[4].buffer(),  m_cntHitsPerLayer[5].buffer(),
        m_cntHitsPerLayer[6].buffer(), m_cntHitsPerLayer[7].buffer(),  m_cntHitsPerLayer[8].buffer(),
        m_cntHitsPerLayer[9].buffer(), m_cntHitsPerLayer[10].buffer(), m_cntHitsPerLayer[11].buffer() };
    std::array<SCbuf, LHCb::Detector::FT::nLayersTotal> bufXPerLayer = {
        m_cntXPerLayer[0].buffer(), m_cntXPerLayer[1].buffer(),  m_cntXPerLayer[2].buffer(),
        m_cntXPerLayer[3].buffer(), m_cntXPerLayer[4].buffer(),  m_cntXPerLayer[5].buffer(),
        m_cntXPerLayer[6].buffer(), m_cntXPerLayer[7].buffer(),  m_cntXPerLayer[8].buffer(),
        m_cntXPerLayer[9].buffer(), m_cntXPerLayer[10].buffer(), m_cntXPerLayer[11].buffer() };
    bufTotalHits += clusters.size();

    for ( auto iZone : hitzones ) {

      hits.setZoneIndex( iZone, hits.size() );
      const auto globalLayerIdx = iZone / LHCb::Detector::FT::nZones;

      if ( m_layerMasks[globalLayerIdx] ) {
        hits.appendColumn( std::numeric_limits<uint8_t>::max(), 1.e9f, {}, {} );
        continue;
      }
      for ( unsigned int localQuarterIdx{ 0 };
            localQuarterIdx < LHCb::Detector::FT::nQuarters / LHCb::Detector::FT::nZones; ++localQuarterIdx ) {
        const auto globalQuarterIdx = bit_cast<unsigned>( iZone * LHCb::Detector::FT::nZones + localQuarterIdx );
        bufHitsPerLayer[globalLayerIdx] += clusters.range( globalQuarterIdx ).size();
        for ( const auto& clus : clusters.range( globalQuarterIdx ) ) {
          const auto id = clus.channelID();
          assert( clus.pseudoSize() < 9 && "Pseudosize of cluster is > 8. Out of range." );
          const auto [x0_orig, yMin, yMax, z0] = cache.calculateXYZFromChannel( id, clus.fractionBit() );
          bufXPerLayer[globalQuarterIdx / LHCb::Detector::FT::nQuarters] += x0_orig;
          float      x0    = x0_orig;
          const auto index = id.globalMatID();

          if ( m_applyMatContractionCalibration ) {
            if ( !cache.matContractionParameterVector[index].empty() ) {
              x0 = cache.calculateCalibratedX( id, x0_orig );
              if ( msgLevel( MSG::DEBUG ) ) {
                debug() << "x0 before contraction correction: " << x0_orig
                        << ", x0 after contraction correction: " << x0 << endmsg;
              }
            } else {
              ++m_noCalib;
            }
          }
          hits.appendColumn( globalLayerIdx, x0,
                             { invClusRes2[clus.pseudoSize()], cache.dzdy[index], cache.dxdy[index], z0 },
                             { id, yMin, yMax } );
        }
      }
      // add a large number at the end of x hits for each zone to stop binary search before end
      hits.appendColumn( std::numeric_limits<uint8_t>::max(), 1.e9f, {}, {} );
      assert( [&] {
        const auto startIndex = hits.getZoneIndex( iZone );
        const auto zoneView   = hits.view_x_values().subspan( startIndex );
        return std::is_sorted( zoneView.begin(), zoneView.end() );
      }() && "SciFi hits not sorted by x position." );
    }
    /** the start index is given by zonesIndexes[zone] but the end index is given by
     * zoneIndexes[zone+2] because upper and lower zones are adjacent in the container
     * so when asking for start and end of zone 22, 24 gives the beginning of the first
     * upper zone which is the end of zone 22
     */
    hits.setZoneIndex( LHCb::Detector::FT::nZonesTotal, hits.getZoneIndex( xu[0] ) );
    // so when asking for lastZone, lastZone+2 gives the very end of the container
    hits.setZoneIndex( LHCb::Detector::FT::nZonesTotal + 1, hits.size() );
    // avoid FPEs
    for ( unsigned i{ 0 }; i < SIMDWrapper::best::types::size; ++i ) {
      hits.appendColumn( std::numeric_limits<uint8_t>::max(), 1.e9f, {}, {} );
    }
    return hits;
  }
} // namespace LHCb::Pr::FT
