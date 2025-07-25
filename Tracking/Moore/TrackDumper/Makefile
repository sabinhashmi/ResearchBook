build_tool=cmake

DEVTOOLS_DATADIR := /cvmfs/lhcb.cern.ch/lib/var/lib/LbEnv/3588/stable/linux-64/lib/python3.12/site-packages/LbDevTools/data

ifeq ($(wildcard build.conf),build.conf)
include build.conf
export LCG_VERSION=$(lcg_version)
endif


# Make sure BINARY_TAG and CMTCONFIG are set and consistent.
# If one is not set, take the value from the other.
# If none is set, use the default platform from build.conf.
ifeq ($(BINARY_TAG)$(CMTCONFIG),)
  ifeq ($(CMTCONFIG),)
    BINARY_TAG := $(platform)
  else
    BINARY_TAG := $(CMTCONFIG)
  endif
  export BINARY_TAG := $(BINARY_TAG)
endif
ifeq ($(CMTCONFIG),)
  CMTCONFIG := $(BINARY_TAG)
  export CMTCONFIG := $(CMTCONFIG)
endif
ifneq ($(BINARY_TAG),$(CMTCONFIG))
  $(error Invalid environment: inconsistent values for BINARY_TAG and CMTCONFIG)
endif

include $(DEVTOOLS_DATADIR)/Makefile-cmake.mk
