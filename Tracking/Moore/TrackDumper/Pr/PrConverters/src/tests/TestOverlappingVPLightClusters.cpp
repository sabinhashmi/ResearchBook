/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "Event/VPLightCluster.h"
#include "GaudiKernel/Point3DTypes.h"
#include "LHCbAlgs/Consumer.h"

#include "VPDet/DeVP.h"
namespace LHCb {
  /**
   *  Tests to see whether all of the clusters in the VPLightClusters container
   *  are indeed present in another. Also checks the consistency
   *  of its content.
   *
   *  If the OutputLevel is set to VERBOSE, all individual missing hits
   *  are explicitly printed in the stdout, along with the local coordinates.
   *
   *  @author Laurent Dufour
   */
  struct TestOverlappingVPLightClusters
      : Algorithm::Consumer<void( VPLightClusters const&, VPLightClusters const&, DeVP const& ),
                            Algorithm::Traits::usesConditions<DeVP>> {

    TestOverlappingVPLightClusters( const std::string& name, ISvcLocator* pSvcLocator )
        : Consumer( name, pSvcLocator,
                    { KeyValue{ "VPLightClusters_A", "" }, KeyValue{ "VPLightClusters_B", "" },
                      KeyValue{ "DeVPLocation", LHCb::Det::VP::det_path } } ) {}
    void operator()( VPLightClusters const&, VPLightClusters const&, DeVP const& ) const override;

    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_missed_in_B{ this, "Missing clusters in [B]" };
    mutable Gaudi::Accumulators::Counter<>                m_found_in_B{ this, "Found clusters in [B]" };

    Gaudi::Property<float> m_tolerance{ this, "Tolerance", 0.001,
                                        "Tolerance (in mm) for coordinates (default: 1 micron)." };

    bool is_same_cluster( LHCb::VPLightCluster const& vplc_a, LHCb::VPLightCluster const& vplc_b,
                          DeVP const& devp ) const {
      // This function does not just use the vplc_a == vplc_b, as verbose
      // information is given regarding where the check fails
      if ( vplc_a.channelID() != vplc_b.channelID() ) return false;

      // now the things that should be  equal
      if ( vplc_a.xfraction() != vplc_b.xfraction() || vplc_a.yfraction() != vplc_b.yfraction() ) {
        if ( msgLevel( MSG::VERBOSE ) ) {
          verbose() << "Clusters with same channelID have different x/y fraction" << endmsg;
          verbose() << "[A] (fx,fy) = " << (unsigned int)vplc_a.xfraction() << "," << (unsigned int)vplc_a.yfraction()
                    << endmsg;
          verbose() << "[B] (fx,fy) = " << (unsigned int)vplc_b.xfraction() << "," << (unsigned int)vplc_b.yfraction()
                    << endmsg;
        }

        return false;
      }

      if ( fabs( vplc_a.x() - vplc_b.x() ) > m_tolerance.value() ||
           fabs( vplc_a.y() - vplc_b.y() ) > m_tolerance.value() ||
           fabs( vplc_a.z() - vplc_b.z() ) > m_tolerance.value() ) {
        const DeVPSensor& sensor    = devp.sensor( vplc_a.channelID() );
        const auto local_position_a = sensor.globalToLocal( Gaudi::XYZPoint( vplc_a.x(), vplc_a.y(), vplc_a.z() ) );
        const auto local_position_b = sensor.globalToLocal( Gaudi::XYZPoint( vplc_b.x(), vplc_b.y(), vplc_b.z() ) );

        if ( msgLevel( MSG::VERBOSE ) ) {
          verbose() << "Position different between two clusters of the same channelID." << endmsg;
          verbose() << "[A] (x,y,z) = " << vplc_a.x() << "," << vplc_a.y() << "," << vplc_a.z();
          verbose() << ", local: (x,y,z) = " << local_position_a.X() << ", " << local_position_a.Y() << ", "
                    << local_position_a.Z() << endmsg;

          verbose() << "[B] (x,y,z) = " << vplc_b.x() << "," << vplc_b.y() << "," << vplc_b.z();
          verbose() << ", local: (x,y,z) = " << local_position_b.X() << ", " << local_position_b.Y() << ", "
                    << local_position_b.Z() << endmsg;
        }
        return false;
      }

      return true;
    }
  };
} // namespace LHCb

DECLARE_COMPONENT_WITH_ID( LHCb::TestOverlappingVPLightClusters, "TestOverlappingVPLightClusters" )

void LHCb::TestOverlappingVPLightClusters::operator()( VPLightClusters const& vplcs_a, VPLightClusters const& vplcs_b,
                                                       DeVP const& devp ) const {

  // Tests for the presence of the VPLightClusters in the Container
  // First test if everything of A is inside of B
  for ( const auto& vplc_a : vplcs_a ) {
    auto it = std::find_if( vplcs_b.begin(), vplcs_b.end(),
                            [&]( const VPLightCluster& vplc_b ) { return is_same_cluster( vplc_a, vplc_b, devp ); } );

    if ( it == vplcs_b.end() ) {
      ++m_missed_in_B;
    } else {
      ++m_found_in_B;
    }
  }
}
