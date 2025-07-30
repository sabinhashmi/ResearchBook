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

#include "Event/FTLiteCluster.h"
#include "Event/PrHits.h"
#include "Event/UTHitCluster.h"
#include "LHCbAlgs/EmptyProducer.h"
#include "PrKernel/UTHitHandler.h"

/** @class ParticlesEmptyProducer
 * @brief dummy producer of an empty container of particles
 */

DECLARE_COMPONENT_WITH_ID( EmptyProducer<LHCb::FTLiteCluster::FTLiteClusters>, "FTRawBankDecoderEmptyProducer" )
DECLARE_COMPONENT_WITH_ID( EmptyProducer<LHCb::UTHitClusters>, "PrStoreUTHitClustersEmptyProducer" )
DECLARE_COMPONENT_WITH_ID( EmptyProducer<UT::HitHandler>, "PrStoreUTHitEmptyProducer" )
DECLARE_COMPONENT_WITH_ID( EmptyProducer<LHCb::Pr::Hits<LHCb::Pr::HitType::UT>>, "PrStorePrUTHitsEmptyProducer" )
