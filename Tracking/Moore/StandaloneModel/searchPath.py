from LbEnv.ProjectEnv.options import (
    EnvSearchPathEntry,
    LHCbDevPathEntry,
    NightlyPathEntry,
    SearchPath,
    SearchPathEntry,
)

path = SearchPath([EnvSearchPathEntry('CMAKE_PREFIX_PATH', '/cvmfs/lhcb.cern.ch/lib/lhcb:/cvmfs/lhcb.cern.ch/lib/lcg/releases:/cvmfs/lhcb.cern.ch/lib/lcg/app/releases:/cvmfs/lhcb.cern.ch/lib/lcg/external:/cvmfs/lhcb.cern.ch/lib/contrib:/cvmfs/lhcb.cern.ch/lib/var/lib/LbEnv/3588/stable/linux-64/lib/python3.12/site-packages/LbDevTools/data/cmake'), EnvSearchPathEntry('CMTPROJECTPATH', ''), EnvSearchPathEntry('LHCBPROJECTPATH', '/cvmfs/lhcb.cern.ch/lib/lhcb')])
