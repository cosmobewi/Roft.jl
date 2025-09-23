.PHONY: init

init:
	mkdir -p /opt/DataRelease /opt/H0LiCOW /opt/CCcovariance /opt/Capse /opt/trained_emu /opt/planck
	if [ ! -d /opt/DataRelease/.git ]; then git clone https://github.com/PantheonPlusSH0ES/DataRelease.git /opt/DataRelease; fi
	if [ ! -d /opt/H0LiCOW/.git ]; then git clone https://github.com/shsuyu/H0LiCOW-public.git /opt/H0LiCOW; fi
	if [ ! -d /opt/CCcovariance/.git ]; then git clone https://gitlab.com/mmoresco/CCcovariance.git /opt/CCcovariance; fi
	if [ ! -d /opt/Capse/.git ]; then git clone https://github.com/CosmologicalEmulators/Capse.jl.git /opt/Capse; fi
	if [ ! -d /opt/planck/COM_Likelihood_Data-baseline_R3.00 ]; then cd /opt/planck && wget -O planck.tar.gz "https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID=151902" && tar -xzf planck.tar.gz && rm planck.tar.gz; fi
	if [ ! -d /opt/trained_emu/TT ]; then cd /opt && wget -O trained_emu.tar.gz "https://zenodo.org/records/17115001/files/trained_emu.tar.gz?download=1" && tar -xzf trained_emu.tar.gz && rm trained_emu.tar.gz; fi
