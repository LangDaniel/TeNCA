from pathlib import Path
import synapseclient
import synapseutils

syn = synapseclient.Synapse()
syn.login()

dst_folder = Path('/mnt/IML-Proj/public_datasets/MAMA-MIA')
if not dst_folder.exists():
    dst_folder.mkdir()

files = synapseutils.syncFromSynapse(syn, 'syn60868042', path=str(dst_folder))
